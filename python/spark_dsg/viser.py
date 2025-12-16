"""Entry point for visualizing scene graph."""

import enum
import functools
import itertools
from dataclasses import dataclass

import numpy as np
import trimesh
import viser
import viser.transforms as vtf
import io
import json
import logging
import base64
import time
import threading
from pathlib import Path
from PIL import Image, ImageDraw

import spark_dsg as dsg


class ColorMode(enum.Enum):
    LAYER = "layer"
    ID = "id"
    LABEL = "label"
    PARENT = "parent"


def _layer_name(layer_key):
    return f"layer_{layer_key.layer}p{layer_key.partition}"


def color_from_label(G, node, default=None):
    if not isinstance(node.attributes, dsg.SemanticNodeAttributes):
        return default or dsg.Color()

    return dsg.distinct_150_color(node.attributes.semantic_label)


def color_from_id(G, node):
    return dsg.colorbrewer_color(node.id.category_id)


def color_from_parent(G, node, parent_func, default=None):
    if not node.has_parent():
        return default or dsg.Color()

    return parent_func(G, G.get_node(node.get_parent()))


def color_from_layer(G, node):
    return dsg.rainbow_color(node.layer.layer)


def colormap_from_modes(key_to_mode, default_colors=None):
    colormap = {}
    for layer_key, mode in key_to_mode.items():
        default = default_colors.get(layer_key) if default_colors is not None else None
        if mode == ColorMode.ID:
            colormap[layer_key] = color_from_id
        elif mode == ColorMode.LABEL:
            colormap[layer_key] = functools.partial(color_from_label, default=default)
        elif mode == ColorMode.PARENT:
            colormap[layer_key] = functools.partial(
                color_from_parent,
                parent_func=lambda G, x: colormap[x.layer](G, x),
                default=default,
            )
        else:
            colormap[layer_key] = color_from_layer

    return colormap


class FlatGraphView:
    def __init__(self, G):
        self._pos = {}
        self._lookup = {}
        self._edges = {}
        self._interlayer_edges = {}

        for layer in itertools.chain(G.layers, G.layer_partitions):
            if layer.num_nodes() == 0:
                continue

            pos = np.zeros((layer.num_nodes(), 3))
            for idx, node in enumerate(layer.nodes):
                pos[idx, :] = node.attributes.position
                self._lookup[node.id.value] = (idx, layer.key)

            edge_tensor = np.zeros((layer.num_edges(), 2), dtype=np.int64)
            for idx, edge in enumerate(layer.edges):
                edge_tensor[idx, 0] = self._lookup[edge.source][0]
                edge_tensor[idx, 1] = self._lookup[edge.target][0]

            self._pos[layer.key] = pos
            self._edges[layer.key] = edge_tensor

        for edge in G.interlayer_edges:
            source_idx, source_layer = self._lookup[edge.source]
            target_idx, target_layer = self._lookup[edge.target]

            # swap indices to enforce ordering
            if source_layer > target_layer:
                source_layer, target_layer = target_layer, source_layer
                source_idx, target_idx = target_idx, source_idx

            if source_layer not in self._interlayer_edges:
                self._interlayer_edges[source_layer] = {target_layer: []}

            if target_layer not in self._interlayer_edges[source_layer]:
                self._interlayer_edges[source_layer][target_layer] = []

            self._interlayer_edges[source_layer][target_layer].append(
                [source_idx, target_idx]
            )

        self._interlayer_edges = {
            s: {t: np.array(edges, dtype=np.int64) for t, edges in c.items()}
            for s, c in self._interlayer_edges.items()
        }

    def pos(self, layer_key, height=None):
        if layer_key not in self._pos:
            return None

        pos = self._pos[layer_key].copy()
        if height:
            pos[:, 2] += height

        return pos

    @property
    def edges(self):
        return self._interlayer_edges

    def layer_edges(self, layer_key):
        return self._edges.get(layer_key)


class ObjectManager:
    def __init__(self, server, G, image_root=None):
        self._server = server
        self._G = G
        self._image_root = Path(image_root) if image_root else None
        
        # Create image container first so it appears at the top
        self._image_container = server.gui.add_folder("Selected Image")

        # Hacks for wider modals
        # Viser uses Mantine UI. We can inject global styles via markdown.
        server.gui.add_markdown(
            "<style>"
            ".mantine-Modal-content { max-width: 95vw !important; width: auto !important; }"
            ".mantine-Modal-inner { width: 100% !important; padding-left: 0 !important; padding-right: 0 !important; }"
            "</style>"
        )

        self._folder = server.gui.add_folder("Object Browser")
        with self._folder:
            self._object_dropdown = server.gui.add_dropdown(
                "Object ID", 
                options=["None"], 
                initial_value="None"
            )
            self._jump_button = server.gui.add_button("Jump to Object")
            self._toggle_mesh = server.gui.add_checkbox("Show Object Mesh", initial_value=True)
            self._toggle_bbox_2d = server.gui.add_checkbox("Show 2D BBox", initial_value=True)
            self._show_mask = server.gui.add_checkbox("Show Mask", initial_value=False)
            self._show_3d_image = server.gui.add_checkbox("Show 3D Image", initial_value=True)
            self._maximize_2d = server.gui.add_checkbox("Maximize 2D Image", initial_value=False)
            
            self._image_handle_2d = None
            self._image_handle_3d = None
            self._image_label_3d = None
            self._mesh_handle = None
            
            # Gallery Controls
            self._image_slider = server.gui.add_slider(
                "Frame Index", min=0, max=1, step=1, initial_value=0, visible=False
            )
            self._maximize_btn = server.gui.add_button("Fullscreen Image", icon=viser.Icon.ZOOM_IN, visible=False)
            
            # Playback Controls placed in a row
            with server.gui.add_folder("Playback Controls", visible=False) as self._playback_folder:
                 self._play_button = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
                 self._pause_button = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE, visible=False)
                 self._fps_number = server.gui.add_number("FPS", initial_value=10.0, min=1.0, max=60.0)

        self._current_node = None
        self._node_map = {} # label -> node_id
        self._current_images = [] # List of (image_path, meta_path, mask_path)
        self._current_img_array = None # Cache for modal
        self._playing = False
        self._playback_thread = None
        
        # Populate dropdown
        self._update_dropdown()
        
        # Event callbacks
        self._object_dropdown.on_update(self._on_object_select)
        self._jump_button.on_click(self._on_jump)
        self._toggle_mesh.on_update(self._on_view_update)
        self._toggle_bbox_2d.on_update(self._on_view_update)
        self._show_mask.on_update(self._on_view_update)
        self._show_3d_image.on_update(self._on_view_update)
        self._maximize_2d.on_update(self._on_view_update)
        
        self._image_slider.on_update(self._on_slider_update)
        self._maximize_btn.on_click(self._on_maximize_click)
        self._play_button.on_click(self._on_play)
        self._pause_button.on_click(self._on_pause)

        # Start playback thread
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()

        if self._image_root:
            print(f"ObjectManager initialized with image_root: {self._image_root}")

    def _update_dropdown(self):
        options = ["None"]
        self._node_map = {}
        
        object_nodes = []
        for layer in self._G.layers:
            for node in layer.nodes:
                # Check if it looks like an object (has bounding box or image folder)
                if hasattr(node.attributes, "bounding_box") or hasattr(node.attributes, "image_folder"):
                     object_nodes.append(node)
                     
        # Sort by ID
        object_nodes.sort(key=lambda x: x.id.value)
        
        for node in object_nodes:
            label = f"{node.id}"
            if hasattr(node.attributes, "name") and node.attributes.name:
                label += f" ({node.attributes.name})"
            options.append(label)
            self._node_map[label] = node.id.value

        self._object_dropdown.options = options

    def handle_click(self, node):
        # Find label for node
        target_val = node.id.value
        found_label = "None"
        for label, val in self._node_map.items():
            if val == target_val:
                found_label = label
                break
        
        if found_label != "None":
            # Only update if different to avoid flicker
            if self._object_dropdown.value != found_label:
                self._object_dropdown.value = found_label
            
            # Use current selection logic
            self._current_node = node
            self._update_selection()

    def _playback_loop(self):
        while True:
            if self._playing and self._current_images:
                try:
                    current_idx = self._image_slider.value
                    max_idx = len(self._current_images) - 1
                    
                    if max_idx > 0:
                        next_idx = (current_idx + 1) % (max_idx + 1)
                        self._image_slider.value = next_idx
                        
                    sleep_time = 1.0 / self._fps_number.value
                    time.sleep(sleep_time)
                except Exception as e:
                    print(f"Playback error: {e}")
                    self._playing = False
            else:
                time.sleep(0.1)

    def _on_slider_update(self, event):
        # Only update if image changed
        if not self._current_node:
             return
        self._update_image()

    def _on_play(self, event):
        self._playing = True
        self._play_button.visible = False
        self._pause_button.visible = True

    def _on_pause(self, event):
        self._playing = False
        self._play_button.visible = True
        self._pause_button.visible = False

    def _on_maximize_click(self, event):
            # Helper to update image based on zoom/pan/frame
            def _update_view():
                if not self._current_images:
                     return
                
                # Get current raw image (re-read or cache?)
                # We cache _current_img_array but that is for the SIDE PANEL (potentially with bbox/mask burned in?)
                # Actually _update_image updates _current_img_array with the *final* 2D image (mask+bbox).
                # So we can use it.
                
                # But wait, if we change frame via slider in modal, _current_img_array updates?
                # Yes, because we link sliders.
                
                # Crop logic
                pil_img = Image.fromarray(self._current_img_array)
                w, h = pil_img.size
                
                zoom = self._modal_zoom.value
                center_x = w / 2 * (1 + self._modal_pan_x.value)
                center_y = h / 2 * (1 + self._modal_pan_y.value)
                
                crop_w = w / zoom
                crop_h = h / zoom
                
                left = center_x - crop_w / 2
                top = center_y - crop_h / 2
                right = center_x + crop_w / 2
                bottom = center_y + crop_h / 2
                
                # Clamping? PIL crop handles partially out of bounds?
                # Better to keep aspect ratio or allow free pan?
                # Let's clean up coordinate math:
                # Pan 0,0 is center. -1 is left edge touches center? No.
                # Let's map Pan -1..1 to moving the center across the image width/height?
                
                # Simple Clamp
                left = max(0, min(left, w - crop_w))
                top = max(0, min(top, h - crop_h))
                
                pil_crop = pil_img.crop((left, top, left + crop_w, top + crop_h))
                
                # Upscale to target
                target_width = 2000
                scale = max(1.0, target_width / float(crop_w))
                new_size = (int(crop_w * scale), int(crop_h * scale))
                
                pil_crop = pil_crop.resize(new_size, Image.NEAREST)
                upscaled_arr = np.array(pil_crop)
                
                self._modal_image_handle.image = upscaled_arr

            with event.client.add_modal("Fullscreen View") as modal:
                # Add Image natively (placeholder, updated immediately)
                self._modal_image_handle = event.client.gui.add_image(
                    self._current_img_array, # temporary
                    format="jpeg",
                )
                
                # Add Controls to Modal
                with event.client.gui.add_folder("Controls"):
                    self._modal_slider_handle = event.client.gui.add_slider(
                        "Frame",
                        min=0,
                        max=len(self._current_images) - 1,
                        step=1,
                        initial_value=self._image_slider.value
                    )
                    
                    self._modal_zoom = event.client.gui.add_slider("Zoom", min=1.0, max=10.0, step=0.1, initial_value=1.0)
                    self._modal_pan_x = event.client.gui.add_slider("Pan X", min=-1.0, max=1.0, step=0.01, initial_value=0.0)
                    self._modal_pan_y = event.client.gui.add_slider("Pan Y", min=-1.0, max=1.0, step=0.01, initial_value=0.0)
                    
                    play_btn = event.client.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
                    pause_btn = event.client.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)
                    close_btn = event.client.gui.add_button("Close", icon=viser.Icon.X)

                    # Link controls
                    self._modal_slider_handle.on_update(lambda _: setattr(self._image_slider, 'value', self._modal_slider_handle.value))
                    
                    # Update view on zoom/pan
                    self._modal_zoom.on_update(lambda _: _update_view())
                    self._modal_pan_x.on_update(lambda _: _update_view())
                    self._modal_pan_y.on_update(lambda _: _update_view())
                    
                    play_btn.on_click(lambda _: self._on_play(None))
                    pause_btn.on_click(lambda _: self._on_pause(None))
                    close_btn.on_click(lambda _: modal.close())
                    
                    # Store update function for external sync?
                    self._modal_update_func = _update_view
                    
                    # Initial update
                    _update_view()

    def _sync_modal(self):
        pass

    def _on_object_select(self, event):
        val = self._object_dropdown.value
        if val == "None":
            self._current_node = None
            self._clear_visuals()
            self._playing = False
            self._image_slider.visible = False
            self._playback_folder.visible = False
            self._maximize_btn.visible = False
            return

        node_id = self._node_map[val]
        self._current_node = self._G.get_node(node_id)
        self._update_selection()

    def _on_jump(self, event):
        if self._current_node and event.client:
            pos = self._current_node.attributes.position
            
            # If 3D image is shown, look at it
            if self._show_3d_image.value and self._image_handle_3d is not None:
                # Get position from handle if possible
                try:
                    target_pos = self._image_handle_3d.position
                    # Image is in XZ plane facing +Y (due to -90 X rotation)
                    # So we want to look at it from +Y direction
                    # target_pos is the center of the image
                    
                    event.client.camera.look_at = target_pos
                    # Stand back in Y
                    # Zoom out more (User request: "zoom out a more")
                    # Previous: 3.0. New: 6.0
                    event.client.camera.position = target_pos + np.array([0.0, 6.0, 0.0])
                    return
                except:
                    pass
            
            # Fallback to oblique view
            event.client.camera.look_at = pos
            # Zoom out a bit
            event.client.camera.position = pos + np.array([-5.0, -5.0, 5.0])

    def _on_view_update(self, event):
        self._update_selection()
        
    def _clear_visuals(self):
        if self._image_handle_2d:
            self._image_handle_2d.remove()
            self._image_handle_2d = None
        if self._image_handle_3d:
            self._image_handle_3d.remove()
            self._image_handle_3d = None
        if self._image_label_3d:
            self._image_label_3d.remove()
            self._image_label_3d = None
        if self._mesh_handle:
            self._mesh_handle.remove()
            self._mesh_handle = None

    def _update_selection(self):
        self._clear_visuals()
        if not self._current_node:
            return

        # Find images first to populate slider
        self._current_images = self._find_all_images(self._current_node)
        
        if len(self._current_images) > 1:
            self._image_slider.max = len(self._current_images) - 1
            self._image_slider.visible = True
            self._playback_folder.visible = True
            # Don't reset slider if playing/scrolling unless out of bounds?
            # Creating a new node selection usually resets.
            # self._image_slider.value = 0 # Optional: Reset to 0 on new object
        else:
            self._image_slider.visible = False
            self._playback_folder.visible = False
            self._playing = False
            self._image_slider.value = 0
            
        # Always show maximize if images exist
        if self._current_images:
            self._maximize_btn.visible = True
        else:
            self._maximize_btn.visible = False
            
        self._update_image()
        self._update_mesh()

    def _update_mesh(self):
        if not self._toggle_mesh.value:
            return
            
        # Check for mesh in attributes
        # KhronosObjectAttributes has 'mesh' method returning Mesh*
        if hasattr(self._current_node.attributes, "mesh"):
            try:
                mesh_ptr = self._current_node.attributes.mesh()
                if not mesh_ptr or mesh_ptr.empty():
                    return
                    
                vertices = mesh_ptr.get_vertices() # 6xN or 3xN? 
                faces = mesh_ptr.get_faces() # 3xM
                
                # vertices usually 3xN (pos) + colors... 
                # wrapper says: vertex_colors=vertices[3:, :].T
                
                if vertices.shape[0] >= 3:
                    v = vertices[:3, :].T
                    f = faces.T
                    
                    colors = None
                    if vertices.shape[0] >= 6:
                        colors = vertices[3:6, :].T
                        
                    trimesh_mesh = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=colors)
                    
                    self._mesh_handle = self._server.scene.add_mesh_trimesh(
                        f"/object_mesh_{self._current_node.id}",
                        trimesh_mesh
                    )
            except Exception as e:
                print(f"Error loading object mesh: {e}")

    def _get_object_label(self, node):
        # Try to get semantic category from labelspace
        try:
            # We need to find which layer this node belongs to to get labelspace
            # NodeId has layer info? node.layer?
            # If not, we search.
            target_layer_id = None
            target_prefix = None
            
            # Helper to check layers
            for layer in self._G.layers:
                if layer.has_node(node.id):
                    target_layer_id = layer.key.layer
                    target_partition = layer.key.partition
                    break
            
            if target_layer_id is not None:
                labelspace = self._G.get_labelspace(target_layer_id, target_partition)
                if labelspace:
                    return labelspace.get_node_category(node)
        except Exception as e:
            print(f"Error getting label for {node.id}: {e}")
        
        # Fallback to name attribute
        if hasattr(node.attributes, "name") and node.attributes.name:
            return node.attributes.name
            
        return "Unknown"

    def _update_image(self):
        if not self._current_node:
            return

        image_path = None
        meta_path = None

        if self._current_images:
            idx = int(self._image_slider.value)
            # Ensure valid index
            idx = max(0, min(idx, len(self._current_images) - 1))
            if idx < len(self._current_images):
                image_path, meta_path, mask_path = self._current_images[idx]
        
        if not image_path:
            # Fallback legacy single image search if list empty?
            # Or just return
            return
            
        try:
            # Load Image
            pil_image = Image.open(image_path)
            
            # Apply Mask if requested and available
            if self._show_mask.value and mask_path and mask_path.exists():
                try:
                    mask_img = Image.open(mask_path).convert("L")
                    # Resize mask if needed (should match, but safety)
                    if mask_img.size != pil_image.size:
                        mask_img = mask_img.resize(pil_image.size)
                    
                    # Ensure mask is visible (0-255) even if it's 0-1 class indices
                    mask_img = mask_img.point(lambda p: 255 if p > 0 else 0)
                    
                    # Create a red overlay
                    # We can use the mask as alpha for a solid color layer
                    overlay = Image.new("RGBA", pil_image.size, (0, 0, 255, 0)) # Blue mask? Or user want specific?
                    # Let's use a semi-transparent red or blue for visibility
                    # Red: (255, 0, 0, 100)
                    
                    # Better: Create an overlay image
                    # Where mask > 0, we add color
                    mask_arr = np.array(mask_img)
                    
                    # Convert main to RGBA
                    pil_image = pil_image.convert("RGBA")
                    
                    # Create colored overlay
                    color_layer = Image.new("RGBA", pil_image.size, (255, 0, 0, 100)) # Red tint
                    
                    # Use mask as alpha for the color layer
                    # Adjust mask to be binary 0-255 for alpha
                    # If mask is 0 (bg) -> 0 alpha. If mask > 0 (obj) -> 100 alpha?
                    # The mask image is likely 0 and 255 already or 0 and 1?
                    # Let's assume non-zero is object
                    
                    # Create a composite
                    pil_image = Image.composite(color_layer, pil_image, mask_img)
                    
                except Exception as e:
                    print(f"Error applying mask: {e}")

            # Draw BBox if requested (for 2D only AND 3D if desired)
            # User request: "check boxes for the boudning box work on the image that's displayed in 3D"
            # So we apply it to the base image used for both.
            if self._toggle_bbox_2d.value and meta_path and meta_path.exists():
                self._draw_2d_bbox(pil_image, meta_path)

            # Convert to numpy for viser
            img_array_final = np.array(pil_image.convert("RGB"))
            self._current_img_array = img_array_final # Cache for modal
            
            # Retrieve Object Name
            obj_name = self._get_object_label(self._current_node)

            # 2D Side Panel Image
            idx_str = ""
            if len(self._current_images) > 1:
                idx_str = f" [{int(self._image_slider.value)}/{len(self._current_images)-1}]"
            label_text = f"{obj_name} ({self._current_node.id}){idx_str}"

            if self._image_handle_2d is not None:
                self._image_handle_2d.image = img_array_final
                self._image_handle_2d.label = label_text
            else:
                with self._image_container:
                    self._image_handle_2d = self._server.gui.add_image(
                        img_array_final,
                        label=label_text,
                        format="jpeg"
                    )
            
            # Sync Modal if open
            try:
                if hasattr(self, "_modal_image_handle") and self._modal_image_handle is not None:
                    # Sync Slider if loop updated it
                    if hasattr(self, "_modal_slider_handle") and self._modal_slider_handle is not None:
                        if self._modal_slider_handle.value != int(self._image_slider.value):
                            self._modal_slider_handle.value = int(self._image_slider.value)
                            
                    # Trigger view update (which handles cropping and setting image)
                    if hasattr(self, "_modal_update_func"):
                        self._modal_update_func()
                        
            except Exception:
                    # Handle invalidated handles (modal closed)
                    self._modal_image_handle = None
                    self._modal_slider_handle = None
                    self._modal_update_func = None
            
            # 3D Scene Image
            if self._show_3d_image.value:
                # Position above the object?
                pos = np.array(self._current_node.attributes.position)
                
                # Calculate top of bbox
                z_offset = 0.5 # Default fallback
                if hasattr(self._current_node.attributes, "bounding_box"):
                     bbox = self._current_node.attributes.bounding_box
                     if bbox.is_valid():
                         # world_P_center is the bbox center
                         # dimensions is full width/height/depth
                         # z_max = center.z + dim.z / 2
                         # But wait, position logic?
                         # Usually node position is centroid. 
                         # Let's rely on bbox center + half height
                         center = np.array(bbox.world_P_center)
                         dims = np.array(bbox.dimensions)
                         z_max = center[2] + dims[2] / 2.0
                         
                         # If we want to place image *above* this point
                         # Current pos is node position.
                         # We want image bottom to be at z_max + padding
                         
                         # Image height in world is 'height'
                         # Image center Z = bottom_Z + height/2
                         # So Image Center Z = (z_max + padding) + height/2
                         
                         # Let's calculate rendering dims first
                         width = 2.0
                         if self._maximize_2d.value:
                            width = 5.0
                         height = width * (img_array_final.shape[0] / img_array_final.shape[1])
                         
                         padding = 0.5
                         target_z_center = z_max + padding + (height / 2.0)
                         
                         # Update pos vector
                         # Keep X,Y from node or bbox center? BBox center is safer for object alignment
                         pos[0] = center[0]
                         pos[1] = center[1]
                         pos[2] = target_z_center
                         
                         z_offset = dims[2] * 0.5 + 1.0 # fallback for old logic if needed, but we use absolute calc now

                # ... dimensions ...
                # Recalculated above for Z positioning
                width = 2.0
                if self._maximize_2d.value:
                    width = 5.0
                height = width * (img_array_final.shape[0] / img_array_final.shape[1])

                wxyz = vtf.SO3.from_x_radians(-np.pi/2).wxyz
                # pos is now the CENTER of the image in world space
                
                if self._image_handle_3d is not None:
                    self._image_handle_3d.image = img_array_final
                    self._image_handle_3d.position = pos
                    self._image_handle_3d.render_width = width
                    self._image_handle_3d.render_height = height
                else:
                    self._image_handle_3d = self._server.scene.add_image(
                        f"/image_3d_{self._current_node.id}",
                        image=img_array_final,
                        render_width=width,
                        render_height=height,
                        position=pos, 
                        wxyz=wxyz,
                        format="jpeg"
                    )
                
                # Add 3D Label above image
                # Top of image = pos.z + height/2
                # Label at Top + 0.2
                label_pos = pos + np.array([0, 0, height/2 + 0.2])
                label_txt_3d = f"{obj_name} ({self._current_node.id})"
                
                if self._image_label_3d is not None:
                    self._image_label_3d.text = label_txt_3d
                    self._image_label_3d.position = label_pos
                else:
                    self._image_label_3d = self._server.scene.add_label(
                        f"/image_label_{self._current_node.id}",
                        text=label_txt_3d,
                        position=label_pos
                    )
            
        except Exception as e:
            print(f"Error loading image for {self._current_node.id}: {e}")

    def _find_all_images(self, node):
        # Similar logic to _find_image_paths but returns all of them sorted
        images_list = []
        
        if not hasattr(node.attributes, "image_folder"):
            return images_list
            
        folder = node.attributes.image_folder
        if not folder:
            return images_list
            
        folder_path = Path(folder)
        if self._image_root:
            if not folder_path.is_absolute():
                folder_path = self._image_root / folder_path
            
        if not folder_path.exists():
            # Fallback logic copy-paste from original find_image_paths
             if "temp" in folder_path.parts:
                try:
                    new_path = Path(str(folder_path).replace("/temp/", "/"))
                    parent = folder_path.parent.parent
                    fallback_node = parent / f"O_{node.id.category_id}"
                    if fallback_node.exists():
                        folder_path = fallback_node
                except Exception:
                    pass

        if not folder_path.exists():
            if self._image_root:
                fallback = self._image_root / f"O_{node.id.category_id}"
                if fallback.exists():
                    folder_path = fallback
        
        if not folder_path.exists():
             return images_list

        # Find all RGB images
        images = list(folder_path.glob("*_rgb.jpg")) + list(folder_path.glob("*_rgb.png"))
        if not images:
            return images_list
            
        # Parse timestamps to sort
        # Pattern: frame_{timestamp}_rgb.jpg
        
        parsed_images = []
        for img in images:
            parts = img.name.split('_')
            ts = 0
            if len(parts) >= 3:
                 try:
                    ts = int(parts[1])
                 except:
                    pass
            
            # Meta file
            meta_file = None
            mask_file = None
            if len(parts) >= 3:
                 # Reconstruct base name? 
                 # frame_TS_meta.json
                 meta_name = f"frame_{parts[1]}_meta.json"
                 meta_file = folder_path / meta_name

                 # Mask file
                 mask_name = f"frame_{parts[1]}_mask.png"
                 mask_file = folder_path / mask_name
                 
            parsed_images.append((ts, img, meta_file, mask_file))
            
        # Sort by timestamp
        parsed_images.sort(key=lambda x: x[0])
        
        return [(x[1], x[2], x[3]) for x in parsed_images]

    def _find_image_paths(self, node):
        # Legacy wrapper or just use first of all images
        all_imgs = self._find_all_images(node)
        if all_imgs:
            # Return path/meta only to match signature if used elsewhere, or just full tuple?
            return all_imgs[0][0], all_imgs[0][1]
        return None, None

    def _draw_2d_bbox(self, pil_image, meta_path):
        try:
            with open(meta_path, 'r') as f:
                data = json.load(f)
            
            if "bbox_2d" in data:
                bbox = data["bbox_2d"]
                min_x = bbox.get("min_x")
                min_y = bbox.get("min_y")
                max_x = bbox.get("max_x")
                max_y = bbox.get("max_y")
                
                if all(v is not None for v in [min_x, min_y, max_x, max_y]):
                    draw = ImageDraw.Draw(pil_image)
                    draw.rectangle([min_x, min_y, max_x, max_y], outline="red", width=3)
        except Exception as e:
            print(f"Error reading meta for bbox: {e}")


@dataclass
class LayerConfig:
    node_scale: float = 0.2
    edge_scale: float = 0.1
    draw_nodes: bool = True
    draw_edges: bool = True
    draw_labels: bool = False
    draw_bboxes: bool = True  # New default


DEFAULT_CONFIG = {
    dsg.LayerKey(2): LayerConfig(node_scale=0.25, draw_labels=True),
    dsg.LayerKey(3): LayerConfig(node_scale=0.1),
    dsg.LayerKey(3, 1): LayerConfig(node_scale=0.1),
    dsg.LayerKey(3, 2): LayerConfig(node_scale=0.1),
    dsg.LayerKey(4): LayerConfig(node_scale=0.4, draw_labels=True, draw_bboxes=True),
    dsg.LayerKey(5): LayerConfig(draw_nodes=False),
}

DEFAULT_COLORMODES = {
    dsg.LayerKey(2): ColorMode.LABEL,
    dsg.LayerKey(3): ColorMode.PARENT,
    dsg.LayerKey(3, 1): ColorMode.LABEL,
    dsg.LayerKey(3, 2): ColorMode.LAYER,
    dsg.LayerKey(4): ColorMode.ID,
    dsg.LayerKey(5): ColorMode.ID,
}


@dataclass
class LabelInfo:
    name: str
    text: str
    pos: np.ndarray


class LayerHandle:
    """Viser handles to layer elements and gui settings."""

    def __init__(
        self, server, config, colormap, height, G, layer, view, parent_callback
    ):
        """Add options for layer to viser."""
        self.key = layer.key
        self.name = _layer_name(layer.key)
        if parent_callback:
            self._parent_callback = parent_callback
        else:
            self._parent_callback = lambda: None
        self._server = server
        self._object_manager = None # Will be set if passed


        self._folder = server.gui.add_folder(self.name)
        try:
            self._folder.open = False # Try to collapse if property exists
        except:
            pass
            
        with self._folder:
            # Layer controls
            self._draw_nodes = server.gui.add_checkbox(
                "draw_nodes", initial_value=config.draw_nodes
            )
            self._draw_labels = server.gui.add_checkbox(
                "draw_labels", initial_value=config.draw_labels
            )
            self._draw_edges = server.gui.add_checkbox(
                "draw_edges", initial_value=config.draw_edges
            )
            self._node_scale = server.gui.add_number(
                "node_scale", initial_value=config.node_scale
            )
            self._edge_scale = server.gui.add_number(
                "edge_scale", initial_value=config.edge_scale
            )
            self._draw_bboxes = server.gui.add_checkbox(
                "draw_bboxes", initial_value=config.draw_bboxes
            )


        self._nodes = None
        self._edges = None
        self._bbox_lines = []
        self._bbox_hitboxes = []
        self._label_info = []
        self._label_handles = []
        pos = view.pos(layer.key, height)
        if pos is None:
            return

        colors = np.zeros(pos.shape)
        for idx, node in enumerate(layer.nodes):
            colors[idx] = colormap(G, node).to_float_array()

        self._nodes = server.scene.add_point_cloud(
            f"{self.name}_nodes", pos, colors=colors
        )

        labelspace = G.get_labelspace(self.key.layer, self.key.partition)
        for idx, node in enumerate(layer.nodes):
            text = node.id.str(literal=False)
            if labelspace:
                text += ": " + labelspace.get_node_category(node)

            self._label_info.append(
                LabelInfo(name=f"label_{node.id.str()}", text=text, pos=pos[idx])
            )

        edge_indices = view.layer_edges(layer.key)
        if edge_indices is not None:
            self._edges = server.scene.add_line_segments(
                f"{self.name}_edges",
                pos[edge_indices],
                (0.0, 0.0, 0.0),
            )

        if self._draw_bboxes.value:
            self._update_bboxes(G, layer)


        self._update()
        self._draw_nodes.on_update(lambda _: self._update())
        self._draw_labels.on_update(lambda _: self._update())
        self._draw_edges.on_update(lambda _: self._update())
        self._node_scale.on_update(lambda _: self._update())
        self._edge_scale.on_update(lambda _: self._update())
        self._draw_bboxes.on_update(lambda _: self._update())

    def set_object_manager(self, manager):
        self._object_manager = manager


    @property
    def draw_nodes(self):
        return self._draw_nodes.value

    @property
    def color_mode(self):
        return self._colormode

    def _update(self):
        draw_edges = self._draw_nodes.value and self._draw_edges.value
        draw_labels = self._draw_nodes.value and self._draw_labels.value
        draw_bboxes = self._draw_nodes.value and self._draw_bboxes.value
        labels_drawn = len(self._label_handles) > 0

        self._nodes.visible = self._draw_nodes.value
        self._nodes.point_size = self._node_scale.value
        if self._edges:
            self._edges.visible = draw_edges
            self._edges.line_width = self._edge_scale.value

        if not draw_labels and labels_drawn:
            for x in self._label_handles:
                x.remove()

            self._label_handles = []

        if draw_labels and not labels_drawn:
            self._label_handles = [
                self._server.scene.add_label(x.name, x.text, position=x.pos)
                for x in self._label_info
            ]
            
        # Toggle BBoxes visibility
        for bbox_handle in self._bbox_lines:
            bbox_handle.visible = draw_bboxes
        for hitbox in self._bbox_hitboxes:
            # Hitbox is invisible (opacity=0) but its SceneNode visible toggle must be true to exist
            hitbox.visible = draw_bboxes


        self._parent_callback()

    def _update_bboxes(self, G, layer):
        # Clear existing
        for h in self._bbox_lines:
            h.remove()
        self._bbox_lines = []
        
        for node in layer.nodes:
            if hasattr(node.attributes, "bounding_box"):
                bbox = node.attributes.bounding_box
                # Check for validity?
                # Draw plain box
                # BoundingBox from bindings might have min/max/world_P_center/dimensions
                # Let's try to get corners.
                # Bindings: .corners() -> numpy array? or list of vectors?
                # From scene_graph.cpp: .def("corners", &BoundingBox::corners)
                # It returns Eigen::Matrix<float, 3, 8> probably. 
                # Actually usually vectors.
                
                if not bbox.is_valid():
                    continue

                try:
                    # Bindings return a list of numpy arrays
                    c_list = bbox.corners()
                    corners = np.array(c_list) # Should be (8, 3)
                    
                    if corners.shape == (8, 3):
                        # Viser expects lines as simple segments pairs
                        # BBox wireframe edges (indices into the 8 corners)
                        # Standard order for AABB usually: 
                        # 0: min
                        # 7: max
                        # It depends on how spark_dsg orders them. 
                        # Assuming standard 0-7 labeling or similar.
                        # Let's try all edges for a cube.
                        # 0-1, 0-2, 0-4
                        # 1-3, 1-5
                        # 2-3, 2-6
                        # 3-7
                        # 4-5, 4-6
                        # 5-7
                        # 6-7
                        # This works if corners are ordered bitwise (x,y,z): 000, 001, 010...
                        
                        # However, let's just trace all 12 edges based on bit permutations if unsure, 
                        # OR just use the previous standard list which is robust for linear ordering if consistent.
                        # Previous list:
                        # (0, 1), (1, 2), (2, 3), (3, 0) -> bottom face?
                        # (4, 5), (5, 6), (6, 7), (7, 4) -> top face?
                        # (0, 4), (1, 5), (2, 6), (3, 7) -> vertical connectors
                        
                        edges = [
                            (0, 1), (1, 3), (3, 2), (2, 0),
                            (4, 5), (5, 7), (7, 6), (6, 4),
                            (0, 4), (1, 5), (2, 6), (3, 7)
                        ]
                        # Wait, spark_dsg bitwise order:
                        # 0: ---
                        # 1: --+
                        # 2: -+-
                        # 3: -++
                        # ...
                        # If that's the case:
                        # 0 connects to 1(z), 2(y), 4(x)
                        # 1 connects to 0, 3(y), 5(x)
                        # 2 connects to 0, 3(z), 6(x)
                        # 3 connects to 1, 2, 7(x)
                        # 4 connects to 0, 5(z), 6(y)
                        # 5 connects to 1, 4, 7(y)
                        # 6 connects to 2, 4, 7(z)
                        # 7 connects to 3, 5, 6
                        
                        edges_bitwise = [
                            (0, 1), (0, 2), (0, 4),
                            (1, 3), (1, 5),
                            (2, 3), (2, 6),
                            (3, 7),
                            (4, 5), (4, 6),
                            (5, 7),
                            (6, 7)
                        ]

                        segments = []
                        # Use the bitwise edges as they are most likely for generated corners
                        for start, end in edges_bitwise:
                            segments.append(corners[start])
                            segments.append(corners[end])
                        
                        segments = np.array(segments).reshape(-1, 2, 3)
                        
                        color = dsg.distinct_150_color(node.attributes.semantic_label).to_float_array() if hasattr(node.attributes, "semantic_label") else (1.0, 0.0, 0.0)

                        line_handle = self._server.scene.add_line_segments(
                            f"{self.name}_bbox_{node.id}",
                            segments,
                            color, 
                            line_width=2.0
                        )
                        self._bbox_lines.append(line_handle)
                        
                        # Create Frame/Hitbox (Transparent Mesh)
                        # We use a box mesh with alpha=0 (or 0.01) to capture clicks
                        # Create Hitbox using add_box for better transparency support
                        dim = np.array(bbox.dimensions)
                        center = np.array(bbox.world_P_center)
                        R = np.array(bbox.world_R_center)
                        wxyz = vtf.SO3.from_matrix(R).wxyz
                        
                        hitbox_handle = self._server.scene.add_box(
                             f"{self.name}_hitbox_{node.id}",
                             position=center,
                             dimensions=dim,
                             wxyz=wxyz,
                             color=(0, 0, 0),
                             opacity=0.0, # Invisible but clickable
                        )
                        hitbox_handle.on_click(lambda _, n=node: self._on_node_click(n))
                        self._bbox_hitboxes.append(hitbox_handle)

                except Exception as e:
                     # Fallback if corners() fails or shape is weird
                     print(f"Error drawing bbox for {node.id}: {e}")
                     pass

    def _on_node_click(self, node):
        if self._object_manager:
            self._object_manager.handle_click(node)


    def remove(self):
        if self._nodes:
            self._nodes.remove()
        if self._edges:
            self._edges.remove()

        for x in self._label_handles:
            x.remove()

        self._label_handles = []

        self._draw_nodes.remove()
        self._draw_edges.remove()
        self._node_scale.remove()
        self._edge_scale.remove()
        self._draw_bboxes.remove()
        for x in self._bbox_lines:
            x.remove()
        for x in self._bbox_hitboxes:
            x.remove()
        self._folder.remove()



class GraphHandle:
    """Visualization handles for a scene graph."""

    def __init__(self, server, G, height_scale=5.0, object_manager=None):
        """Draw a scene graph in the visualizer."""
        self._handles = {}
        self._edge_handles = {}
        self._height_scale = height_scale
        self._object_manager = object_manager
        self._edge_scale = server.gui.add_number(
            "interlayer_edge_scale", initial_value=0.1
        )

        color_modes = {}
        for layer in itertools.chain(G.layers, G.layer_partitions):
            color_modes[layer.key] = DEFAULT_COLORMODES.get(layer.key, ColorMode.LAYER)

        colormaps = colormap_from_modes(color_modes)

        view = FlatGraphView(G)
        for layer in itertools.chain(G.layers, G.layer_partitions):
            self._handles[layer.key] = LayerHandle(
                server,
                DEFAULT_CONFIG.get(layer.key, LayerConfig()),
                colormaps[layer.key],
                self._layer_height(layer.key),
                G,
                layer,
                view,
                self._update,
            )
            self._handles[layer.key].set_object_manager(object_manager)


        for source_layer, targets in view.edges.items():
            self._edge_handles[source_layer] = {}
            for target_layer, edge_indices in targets.items():
                source_pos = view.pos(source_layer, self._layer_height(source_layer))
                target_pos = view.pos(target_layer, self._layer_height(target_layer))
                assert source_pos is not None and target_pos is not None

                source_name = _layer_name(source_layer)
                target_name = _layer_name(target_layer)
                pos = np.vstack((source_pos, target_pos))

                edge_indices = edge_indices.copy()
                edge_indices[:, 1] += source_pos.shape[0]

                self._edge_handles[source_layer][target_layer] = (
                    server.scene.add_line_segments(
                        f"{source_name}_to_{target_name}",
                        pos[edge_indices],
                        (0.0, 0.0, 0.0),
                    )
                )

        self._update()
        self._edge_scale.on_update(lambda _: self._update())

    def remove(self):
        """Remove graph elements from the visualizer."""
        self._edge_scale.remove()
        for _, handle in self._handles.items():
            handle.remove()

        self._handles = {}

        for _, handles in self._edge_handles.items():
            for _, handle in handles.items():
                handle.remove()

        self._edge_handles = {}

    def _layer_height(self, layer_key):
        return self._height_scale * layer_key.layer

    def _update(self):
        for source_key, targets in self._edge_handles.items():
            for target_key, handle in targets.items():
                handle.visible = (
                    self._handles[source_key].draw_nodes
                    and self._handles[target_key].draw_nodes
                )
                handle.line_width = self._edge_scale.value


class MeshHandle:
    """Visualizer handle for mesh elements."""

    def __init__(self, server, mesh):
        """Send a mesh to the visualizer."""
        vertices = mesh.get_vertices()
        mesh = trimesh.Trimesh(
            vertices=vertices[:3, :].T,
            faces=mesh.get_faces().T,
            visual=trimesh.visual.ColorVisuals(vertex_colors=vertices[3:, :].T),
        )

        self._mesh_handle = server.scene.add_mesh_trimesh(name="/mesh", mesh=mesh)

    def remove(self):
        """Remove mesh elements from the visualizer."""
        self._mesh_handle.remove()


class ViserRenderer:
    """Rendering interface to Viser client."""

    def __init__(self, ip="localhost", port=8080, clear_at_exit=True, image_root=None):
        self._server = viser.ViserServer(host=ip, port=port)
        self._clear_at_exit = clear_at_exit
        self._mesh_handle = None
        self._graph_handle = None
        self._image_root = image_root
        self._object_manager = None


    def __enter__(self):
        """Enter a context manager."""
        return self

    def __exit__(self, typ, exc, tb):
        """Clean up visualizer on exit if desired."""
        print(f"typ='{typ}' exc='{exc}' tb='{tb}'")
        if self._clear_at_exit:
            self.clear()

        return True

    def draw(self, G, height_scale=5.0):
        self._clear_graph()
        self._object_manager = ObjectManager(self._server, G, self._image_root)
        self._graph_handle = GraphHandle(self._server, G, height_scale=height_scale, object_manager=self._object_manager)

        if G.has_mesh():
            self.draw_mesh(G.mesh)

    def draw_mesh(self, mesh):
        self._clear_mesh()
        self._mesh_handle = MeshHandle(self._server, mesh)

    def clear(self):
        """Remove all graph and mesh elements from the visualizer."""
        self._clear_mesh()
        self._clear_graph()

    def _clear_mesh(self):
        if self._mesh_handle:
            self._mesh_handle.remove()
            self._mesh_handle = None

    def _clear_graph(self):
        if self._graph_handle:
            self._graph_handle.remove()
            self._graph_handle = None
