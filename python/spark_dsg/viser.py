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

# Cache valid for mostly static images on disk
@functools.lru_cache(maxsize=128)
def load_and_process_image(path, max_dim=1024):
    try:
        with Image.open(path) as img:
            # Resize if too large to save bandwidth/memory
            w, h = img.size
            if max(w, h) > max_dim:
                scale = max_dim / float(max(w, h))
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
            
            # Convert to RGB to ensure consistency
            if img.mode != "RGB":
                img = img.convert("RGB")
                
            return np.array(img)
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None


import spark_dsg as dsg


class ColorMode(enum.Enum):
    LAYER = "layer"
    ID = "id"
    LABEL = "label"
    PARENT = "parent"


def _layer_name(layer_key):
    # Static mappings based on spark_dsg bindings
    base_names = {
        2: "Objects",
        3: "Places",
        4: "Rooms",
        5: "Buildings"
    }
    
    # Special handle for known partitions
    if layer_key.layer == 3 and layer_key.partition == 1:
        return "Mesh Places"
        
    base = base_names.get(layer_key.layer, f"Layer {layer_key.layer}")
    
    if layer_key.partition != 0:
        if layer_key.layer == 2:
            return f"Agents ({layer_key.partition})"
        return f"{base} (p{layer_key.partition})"
        
    return base


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
    def __init__(self, server, G, image_root=None, image_folder_prefix=None):
        self._server = server
        self._G = G
        self._image_root = Path(image_root) if image_root else None
        self._image_folder_prefix = Path(image_folder_prefix) if image_folder_prefix else None
        
        # Create image container first so it appears at the top
        self._image_container = server.gui.add_folder("Selected Image")

        # Hacks for wider modals
        # Viser uses Mantine UI. Inject global styles via HTML so <style> is respected.
        server.gui.add_html(
            "<style>"
            # Default modal size (smaller than fullscreen), with small viewport margins.
            ".mantine-Modal-root { --modal-size: 55vw !important; }"

            # Make the modal resizable via corner drag. Important: don't lock width/height
            # with !important, otherwise the browser resize handle can't change it.
            ".mantine-Modal-content {"
            "  flex: none !important;"
            "  flex-basis: auto !important;"
            "  width: var(--modal-size);"
            "  height: 65vh;"
            "  max-width: 95vw !important;"
            "  max-height: 95vh !important;"
            "  min-width: 360px;"
            "  min-height: 320px;"
            "  display: flex;"
            "  flex-direction: column;"
            "  resize: both;"
            "  overflow: hidden;"
            "}"

            ".mantine-Modal-inner { width: 100% !important; padding-left: 0 !important; padding-right: 0 !important; }"
            ".mantine-Modal-body { padding: 0 !important; flex: 1 1 auto !important; overflow: auto !important; }"

            # Ensure image doesn't push controls off-screen.
            ".mantine-Modal-body img { width: 100% !important; height: auto !important; max-height: 42vh !important; object-fit: contain !important; display: block !important; }"
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
            
            # Sidebar Navigation
            # Place small buttons for prev/next
            with server.gui.add_folder("Navigation", visible=True) as nav_folder:
                 self._side_prev_btn = server.gui.add_button("Prev", icon=viser.Icon.ARROW_LEFT)
                 self._side_next_btn = server.gui.add_button("Next", icon=viser.Icon.ARROW_RIGHT)
                 
            self._side_prev_btn.on_click(lambda _: self._step_object(-1))
            self._side_next_btn.on_click(lambda _: self._step_object(1))
            
            self._toggle_mesh = server.gui.add_checkbox("Show Object Mesh", initial_value=True)
            self._toggle_bbox_2d = server.gui.add_checkbox("Show 2D BBox", initial_value=True)
            self._show_mask = server.gui.add_checkbox("Show Mask", initial_value=False)
            self._display_mode = server.gui.add_dropdown("Display Mode", options=["RGB", "Depth", "Both"], initial_value="RGB")
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
                 self._save_gif_btn = server.gui.add_button("Save as GIF", icon=viser.Icon.FILE_DOWNLOAD)

        self._current_node = None
        self._node_map = {} # label -> node_id
        self._current_images = [] # List of (image_path, depth_path, meta_path, mask_path)
        self._current_img_array = None # Cache for modal
        self._playing = False
        self._playback_thread = None
        # Re-entrant because we sometimes update UI from codepaths that already hold the lock.
        self._lock = threading.RLock()
        # Guard to prevent modal slider sync from recursively updating the main slider.
        self._syncing_modal_slider = False

        # Modal-only UI handles (for syncing playback controls).
        self._modal_play_btn = None
        self._modal_pause_btn = None
        self._modal_handle = None
        
        # Populate dropdown
        
        # Populate dropdown
        self._update_dropdown()
        
        # Event callbacks
        self._object_dropdown.on_update(self._on_object_select)
        self._jump_button.on_click(self._on_jump)
        self._toggle_mesh.on_update(self._on_view_update)
        self._toggle_bbox_2d.on_update(self._on_view_update)
        self._show_mask.on_update(self._on_view_update)
        self._display_mode.on_update(self._on_view_update)
        self._show_3d_image.on_update(self._on_view_update)
        self._maximize_2d.on_update(self._on_view_update)
        
        self._image_slider.on_update(self._on_slider_update)
        self._maximize_btn.on_click(self._on_maximize_click)
        self._play_button.on_click(self._on_play)
        self._pause_button.on_click(self._on_pause)
        self._save_gif_btn.on_click(self._on_save_gif)

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
                # Sanitize name to avoid JS issues
                safe_name = str(node.attributes.name).replace('"', '').replace("'", "").replace("\\", "")
                label += f" ({safe_name})"
            options.append(label)
            self._node_map[label] = node.id.value

        self._object_dropdown.options = options

    def handle_click(self, node):
        # Find label for node
        target_val = node.id.value
        found_label = "None"
        with self._lock:
            for label, val in self._node_map.items():
                if val == target_val:
                    found_label = label
                    break
        
        if found_label != "None":
            # Only update if different to avoid flicker
            if self._object_dropdown.value != found_label:
                self._object_dropdown.value = found_label
            
            # Use current selection logic which acquires lock internally
            # But we set _current_node here? No, _on_object_select does.
            # We just change the dropdown, which triggers _on_object_select
            pass

    def _playback_loop(self):
        while True:
            # Snapshot state safely
            should_play = False
            image_count = 0
            
            with self._lock:
                should_play = self._playing
                image_count = len(self._current_images)
            
            if should_play and image_count > 0:
                try:
                    # We need to update slider on main thread usually? 
                    # Viser handles are thread safe for updates generally, but logic needs sync
                    current_idx = self._image_slider.value
                    max_idx = image_count - 1
                    
                    if max_idx > 0:
                        next_idx = (current_idx + 1) % (max_idx + 1)
                        self._image_slider.value = next_idx
                        
                    sleep_time = 1.0 / max(1.0, self._fps_number.value)
                    time.sleep(sleep_time)
                except Exception as e:
                    print(f"Playback error: {e}")
                    with self._lock:
                        self._playing = False
            else:
                time.sleep(0.1)

    def _on_slider_update(self, event):
        # Only update if image changed
        with self._lock:
            if not self._current_node:
                 return
            self._update_image()

    def _on_play(self, event):
        self._playing = True
        self._play_button.visible = False
        self._pause_button.visible = True
        # Sync modal buttons if fullscreen is open.
        try:
            if self._modal_play_btn is not None:
                self._modal_play_btn.visible = False
            if self._modal_pause_btn is not None:
                self._modal_pause_btn.visible = True
        except Exception:
            pass

    def _on_pause(self, event):
        self._playing = False
        self._play_button.visible = True
        self._pause_button.visible = False
        # Sync modal buttons if fullscreen is open.
        try:
            if self._modal_play_btn is not None:
                self._modal_play_btn.visible = True
            if self._modal_pause_btn is not None:
                self._modal_pause_btn.visible = False
        except Exception:
            pass

    def _generate_and_download_gif(self, client, button_handle=None):
        """Helper to generate and download GIF. Shared by main and modal uis."""
        if not self._current_images:
            return

        # Disable button/Update label if handle provided
        if button_handle:
            button_handle.disabled = True
            button_handle.label = "Generating..."
        
        try:
            images = []
            durations = []
            fps = float(self._fps_number.value)
            frame_duration_ms = 1000.0 / fps
            
            # Iterate all frames
            for i in range(len(self._current_images)):
                 img = self._get_composed_image(i)
                 if img:
                     # Resize for reasonable GIF size if too large
                     if img.width > 800:
                         scaling = 800.0 / img.width
                         new_size = (int(img.width * scaling), int(img.height * scaling))
                         img = img.resize(new_size, Image.LANCZOS)
                         
                     images.append(img)
                     durations.append(frame_duration_ms)
            
            if not images:
                return

            # Save to buffer
            buf = io.BytesIO()
            images[0].save(
                buf,
                format="GIF",
                save_all=True,
                append_images=images[1:],
                optimize=True,
                duration=durations,
                loop=0
            )
            buf.seek(0)
            
            # Trigger download
            file_data = buf.getvalue()
            filename = f"sequence_{self._current_node.id}.gif"
            
            client.send_file_download(
                filename,
                file_data
            )
            
        except Exception as e:
            print(f"Error generating GIF: {e}")
        finally:
             if button_handle:
                 button_handle.disabled = False
                 button_handle.label = "Save as GIF"

    def _on_save_gif(self, event):
        """Generate and download a GIF of the current sequence."""
        self._generate_and_download_gif(event.client, self._save_gif_btn)

    def _step_object(self, delta):
        """Cycle through objects in the dropdown."""
        options = self._object_dropdown.options
        if not options:
            return
            
        current_val = self._object_dropdown.value
        try:
            idx = options.index(current_val)
            new_idx = (idx + delta) % len(options)
            self._object_dropdown.value = options[new_idx]
        except ValueError:
            if options:
                self._object_dropdown.value = options[0]

    def _on_maximize_click(self, event):
            # Helper to update image based on zoom/pan/frame
            def _update_view():
                # Check if modal is still open
                if (not self._current_images or 
                    self._modal_image_handle is None or 
                    self._modal_zoom is None or 
                    self._modal_pan_x is None or 
                    self._modal_pan_y is None):
                    return
                
                try:
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
                    
                    # Update dynamic title
                    if hasattr(self, "_modal_title_handle") and self._modal_title_handle is not None and self._current_node:
                        obj_name = self._get_object_label(self._current_node)
                        title_text = f"## {obj_name} ({self._current_node.id})"
                        self._modal_title_handle.content = title_text
                        
                except Exception as e:
                    # Modal was closed or handle became invalid
                    pass

            # If a modal is already open, close it before creating a new one.
            try:
                if self._modal_handle is not None:
                    self._modal_handle.close()
            except Exception:
                pass

            # Get title with object name and class/ID

            # Get title with object name and class/ID
            init_title = ""
            if self._current_node:
                obj_name = self._get_object_label(self._current_node)
                init_title = f"## {obj_name} ({self._current_node.id})"

            with event.client.gui.add_modal("Fullscreen View") as modal:
                self._modal_handle = modal
                
                # Dynamic Title
                self._modal_title_handle = event.client.gui.add_markdown(init_title)

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
                    
                    # Add Visual Controls (Synced)
                    # We initialize with current values
                    m_bbox = event.client.gui.add_checkbox("Show 2D BBox", initial_value=self._toggle_bbox_2d.value)
                    m_mask = event.client.gui.add_checkbox("Show Mask", initial_value=self._show_mask.value)
                    m_mode = event.client.gui.add_dropdown("Display Mode", options=["RGB", "Depth", "Both"], initial_value=self._display_mode.value)
                    
                    # Callbacks to sync to main controls
                    # Updating main controls will trigger _on_view_update -> _update_image -> _modal_update_func
                    
                    def _sync_bbox(_):
                        self._toggle_bbox_2d.value = m_bbox.value
                        
                    def _sync_mask(_):
                         self._show_mask.value = m_mask.value
                         
                    def _sync_mode(_):
                         self._display_mode.value = m_mode.value
                         
                    m_bbox.on_update(_sync_bbox)
                    m_mask.on_update(_sync_mask)
                    m_mode.on_update(_sync_mode)
                    

                    
                    # Navigation Buttons (Placed above GIF/Play for better visibility)
                    with event.client.gui.add_folder("Navigation"):
                        b_prev = event.client.gui.add_button("Prev Object", icon=viser.Icon.ARROW_LEFT)
                        b_next = event.client.gui.add_button("Next Object", icon=viser.Icon.ARROW_RIGHT)
                    
                    b_prev.on_click(lambda _: self._step_object(-1))
                    b_next.on_click(lambda _: self._step_object(1))

                    # GIF Button for Modal
                    m_save_gif = event.client.gui.add_button("Save as GIF", icon=viser.Icon.FILE_DOWNLOAD)
                    
                    def _modal_save_gif(evt):
                        self._generate_and_download_gif(evt.client, m_save_gif)
                        
                    m_save_gif.on_click(_modal_save_gif)


                    
                    # Playback
                    play_btn = event.client.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
                    pause_btn = event.client.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE, visible=False)
                    close_btn = event.client.gui.add_button("Close", icon=viser.Icon.X)

                    # Store handles for syncing with main playback buttons.
                    self._modal_play_btn = play_btn
                    self._modal_pause_btn = pause_btn

                    # Initialize modal buttons to match current playback state.
                    if self._playing:
                        play_btn.visible = False
                        pause_btn.visible = True
                    else:
                        play_btn.visible = True
                        pause_btn.visible = False

                    # Capture handles in local variables for safe callback access
                    modal_slider = self._modal_slider_handle
                    main_slider = self._image_slider
                    
                    # Link modal slider to main slider AND update view
                    def _on_modal_slider_change(_):
                        # If the main UI is syncing the modal slider, don't reflect back.
                        if getattr(self, "_syncing_modal_slider", False):
                            _update_view()
                            return

                        # Update main slider (which triggers _update_image)
                        main_slider.value = modal_slider.value
                        # Also update the modal view immediately
                        _update_view()
                    
                    modal_slider.on_update(_on_modal_slider_change)
                    
                    # Update view on zoom/pan
                    self._modal_zoom.on_update(lambda _: _update_view())
                    self._modal_pan_x.on_update(lambda _: _update_view())
                    self._modal_pan_y.on_update(lambda _: _update_view())
                    
                    # Modal-specific play/pause handlers that manage modal button visibility
                    def _modal_play(_):
                        self._playing = True
                        play_btn.visible = False
                        pause_btn.visible = True

                        # Sync main controls.
                        self._play_button.visible = False
                        self._pause_button.visible = True
                    
                    def _modal_pause(_):
                        self._playing = False
                        play_btn.visible = True
                        pause_btn.visible = False

                        # Sync main controls.
                        self._play_button.visible = True
                        self._pause_button.visible = False
                    
                    play_btn.on_click(_modal_play)
                    pause_btn.on_click(_modal_pause)

                    def _close_modal(_):
                        try:
                            modal.close()
                        finally:
                            # Cleanup after modal closes.
                            self._modal_handle = None
                            self._modal_image_handle = None
                            self._modal_slider_handle = None
                            self._modal_update_func = None
                            self._modal_zoom = None
                            self._modal_pan_x = None
                            self._modal_pan_y = None
                            self._modal_play_btn = None
                            self._modal_pause_btn = None

                    close_btn.on_click(_close_modal)
                    
                    # Store update function for external sync
                    self._modal_update_func = _update_view
                    
                    # Initial update
                    _update_view()

                    # Note: the modal context manager only scopes *where GUI elements are added*.
                    # We keep handles alive until the user clicks our Close button (or until an
                    # update throws and we clear stale handles in the exception handler).

    def _sync_modal(self):
        pass

    def _on_object_select(self, event):
        val = self._object_dropdown.value
        with self._lock:
            if val == "None":
                self._current_node = None
                self._clear_visuals()
                self._playing = False
                self._image_slider.visible = False
                self._playback_folder.visible = False
                self._maximize_btn.visible = False
                return

            node_id = self._node_map.get(val)
            if node_id is None:
                return
                
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
        with self._lock:
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
        # Assumes lock is held by caller
        self._clear_visuals()
        if not self._current_node:
            return

        # Find images first to populate slider
        self._current_images = self._find_all_images(self._current_node)
        
        if len(self._current_images) > 1:
            self._image_slider.max = len(self._current_images) - 1
            self._image_slider.visible = True
            self._playback_folder.visible = True
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

    def _get_composed_image(self, idx):
        """Helper to get a single composed frame (RGB, Depth, or Both) as a PIL image."""
        if not self._current_images or idx >= len(self._current_images):
            return None
            
        image_path, depth_path, meta_path, mask_path = self._current_images[idx]
        
        # If RGB required but missing, return (unless handle depth only mode)
        # We can handle depth-only if image_path is None but depth_path exists? 
        # For now assume RGB always exists if we found it.
        if not image_path:
            return None
            
        try:
            mode = self._display_mode.value
            
            # --- Prepare RGB ---
            pil_rgb = None
            if mode in ["RGB", "Both"] and image_path:
                img_array_base = load_and_process_image(str(image_path), max_dim=1024)
                if img_array_base is not None:
                    pil_rgb = Image.fromarray(img_array_base)
                    
                    # Apply Mask/BBox to RGB
                    self._process_single_image_view(pil_rgb, mask_path, meta_path, original_size=self._get_original_size(image_path))

            # --- Prepare Depth ---
            pil_depth = None
            if mode in ["Depth", "Both"] and depth_path and depth_path.exists():
                try:
                    # Depth is likely 16-bit or just grayscale png.
                    # Load as is
                    with Image.open(depth_path) as d_img:
                        # Convert to numpy array directly to handle all modes (I;16, I, F, L) robustly
                        arr = np.array(d_img)
                        
                        # Handle potential float/int types
                        # If image is float or large integer, we need to normalize.
                        # If it's already uint8 (L), we might still want to stretch contrast if it's dark?
                        # E.g. if max is 50, it looks black.
                        
                        # Robust normalization:
                        # 1. Ignore 0s (often invalid/mask) for min calculation if possible
                        valid_mask = arr > 0
                        
                        # If float or large int, convert to float32 first to avoid overflow/underflow issues during calc
                        arr_f = arr.astype(np.float32)
                        
                        if valid_mask.any():
                            # Use percentiles to robustly find range
                            v_vals = arr_f[valid_mask]
                            d_min = np.percentile(v_vals, 2)
                            d_max = np.percentile(v_vals, 98)
                            
                            # Normalize
                            if d_max > d_min:
                                norm = (arr_f - d_min) / (d_max - d_min)
                                norm = np.clip(norm, 0, 1)
                                arr_u8 = (norm * 255.0).astype(np.uint8)
                            else:
                                arr_u8 = np.full_like(arr_f, 128, dtype=np.uint8)
                                
                            # Force invalid to black
                            arr_u8[~valid_mask] = 0
                        else:
                            arr_u8 = np.zeros_like(arr_f, dtype=np.uint8)
                            
                        pil_depth = Image.fromarray(arr_u8).convert("RGB")
                            
                    # Resize depth to match RGB if both are present
                    if pil_rgb and pil_depth.size != pil_rgb.size:
                        pil_depth = pil_depth.resize(pil_rgb.size, Image.NEAREST)
                        
                    # Apply Mask/BBox to Depth
                    # Use same meta/mask path
                    self._process_single_image_view(pil_depth, mask_path, meta_path, original_size=self._get_original_size(image_path)) # Use RGB orig size for scale ref?
                    
                except Exception as e:
                    print(f"Error loading depth: {e}")

            # --- Combine ---
            final_pil = None
            
            if mode == "RGB":
                final_pil = pil_rgb
            elif mode == "Depth":
                final_pil = pil_depth if pil_depth else pil_rgb # Fallback
            elif mode == "Both":
                if pil_rgb and pil_depth:
                    # Side by side
                    w1, h1 = pil_rgb.size
                    w2, h2 = pil_depth.size
                    # Assume same height (resized above)
                    total_w = w1 + w2
                    max_h = max(h1, h2)
                    
                    final_pil = Image.new("RGB", (total_w, max_h))
                    final_pil.paste(pil_rgb, (0, 0))
                    final_pil.paste(pil_depth, (w1, 0))
                elif pil_rgb:
                    final_pil = pil_rgb
                elif pil_depth:
                     final_pil = pil_depth
            
            return final_pil
            
        except Exception as e:
            print(f"Error composing image: {e}")
            return None


    def _update_image(self):
        # Check lock? Usually called from protected methods.
        if not self._current_node:
            return

        final_pil = None
        if self._current_images:
            idx = int(self._image_slider.value)
            # Ensure valid index
            idx = max(0, min(idx, len(self._current_images) - 1))
            
            final_pil = self._get_composed_image(idx)
        
        if not final_pil:
            return

        # Convert to numpy for viser
        img_array_final = np.array(final_pil)
        self._current_img_array = img_array_final # Cache for modal
            
        try:
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
                if hasattr(self, "_modal_update_func") and self._modal_update_func is not None:
                    # Update modal slider position and refresh the modal image.
                    if hasattr(self, "_modal_slider_handle") and self._modal_slider_handle is not None:
                        current_frame = int(self._image_slider.value)
                        if self._modal_slider_handle.value != current_frame:
                            self._syncing_modal_slider = True
                            try:
                                self._modal_slider_handle.value = current_frame
                            finally:
                                self._syncing_modal_slider = False
                    
                    # Always trigger view update to refresh the modal image with new frame
                    self._modal_update_func()
                        
            except Exception as e:
                    # Handle invalidated handles (modal closed)
                    print(f"Modal sync error: {e}")
                    self._modal_image_handle = None
                    self._modal_slider_handle = None
                    self._modal_update_func = None
                    self._modal_zoom = None
                    self._modal_pan_x = None
                    self._modal_pan_y = None
            
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

        folder_path = None
        if self._image_folder_prefix:
             prefix_path = self._image_folder_prefix / f"O_{node.id.category_id}"
             if prefix_path.exists():
                 folder_path = prefix_path

        if not folder_path:
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
            depth_file = None
            
            if len(parts) >= 3:
                 # Reconstruct base name? 
                 # frame_TS_meta.json
                 ts_str = parts[1]
                 meta_name = f"frame_{ts_str}_meta.json"
                 meta_file = folder_path / meta_name

                 # Mask file
                 mask_name = f"frame_{ts_str}_mask.png"
                 mask_file = folder_path / mask_name
                 
                 # Depth file
                 depth_name = f"frame_{ts_str}_depth.png"
                 depth_file = folder_path / depth_name
                 
            parsed_images.append((ts, img, depth_file, meta_file, mask_file))
            
        # Sort by timestamp
        parsed_images.sort(key=lambda x: x[0])
        
        return [(x[1], x[2], x[3], x[4]) for x in parsed_images]

    def _find_image_paths(self, node):
        # Legacy wrapper or just use first of all images
        all_imgs = self._find_all_images(node)
        if all_imgs:
            # Unpack: path, depth, meta, mask
            return all_imgs[0][0], all_imgs[0][2]
        return None, None

    def _get_original_size(self, path):
         # Just read header to get size
         try:
             with Image.open(path) as img:
                 return img.size
         except:
             return (1, 1)

    def _draw_2d_bbox(self, pil_image, meta_path, original_size=None):
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
                    
                    # Scale if necessary
                    if original_size and original_size != pil_image.size:
                        orig_w, orig_h = original_size
                        cur_w, cur_h = pil_image.size
                        scale_x = cur_w / float(orig_w)
                        scale_y = cur_h / float(orig_h)
                        
                        min_x *= scale_x
                        max_x *= scale_x
                        min_y *= scale_y
                        max_y *= scale_y
                        
                    draw = ImageDraw.Draw(pil_image)
                    draw.rectangle([min_x, min_y, max_x, max_y], outline="red", width=3)
        except Exception as e:
            print(f"Error reading meta for bbox: {e}")

    def _process_single_image_view(self, pil_image, mask_path, meta_path, original_size=None):
        """Helper to apply mask and bbox to a single PIL image in-place."""
        # Apply Mask if requested and available
        if self._show_mask.value and mask_path and mask_path.exists():
            try:
                mask_img = Image.open(mask_path).convert("L")
                
                # Resize mask to match base image
                if mask_img.size != pil_image.size:
                    mask_img = mask_img.resize(pil_image.size, Image.NEAREST)
                
                # Create red overlay
                color_layer = Image.new("RGBA", pil_image.size, (255, 0, 0, 100))
                
                # Composite
                # Ensure base is RGBA
                if pil_image.mode != "RGBA":
                    pil_image_rgba = pil_image.convert("RGBA")
                    pil_image.paste(Image.composite(color_layer, pil_image_rgba, mask_img), (0,0))
                else:
                    # In-place paste not easy with composite returning new, so paste over
                    composite = Image.composite(color_layer, pil_image, mask_img)
                    pil_image.paste(composite, (0, 0))
                
            except Exception as e:
                print(f"Error applying mask: {e}")

        # Draw BBox if requested
        if self._toggle_bbox_2d.value and meta_path and meta_path.exists():
             self._draw_2d_bbox(pil_image, meta_path, original_size=original_size)


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
        self._path_handle = None
        if parent_callback:
            self._parent_callback = parent_callback
        else:
            self._parent_callback = lambda: None
        self._server = server
        self._object_manager = None # Will be set if passed
        self._G = G  # Store graph reference for labelspace access
        self._layer = layer  # Store layer reference


        self._folder = server.gui.add_folder(self.name, expand_by_default=False)
            
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
            self._draw_bbox_labels = server.gui.add_checkbox(
                "draw_bbox_labels", initial_value=False
            )
            
            # Special: Path drawing for Agents
            self._draw_path = None
            if layer.key.layer == 2: # Agents
                 self._draw_path = server.gui.add_checkbox("draw_path", initial_value=False)
                 self._draw_path.on_update(lambda _: self._update())


        self._nodes = None
        self._edges = None
        self._nodes = None
        self._edges = None
        self._bbox_lines_handle = None
        self._bbox_hitbox_handle = None
        self._bbox_id_map = [] # List of nodes corresponding to batched indices
        self._bbox_label_handles = [] # List of label handles for bounding boxes

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
            
        # Draw Path if requested
        if self._path_handle:
            self._path_handle.remove()
            self._path_handle = None
            
        if self._draw_path and self._draw_path.value:
             try:
                 # We need nodes sorted by ID to form a path
                 # layer.nodes is list[SceneGraphNode]
                 sorted_nodes = sorted(layer.nodes, key=lambda n: n.id.value)
                 
                 # Extract positions
                 points = np.array([n.attributes.position for n in sorted_nodes])
                 
                 if len(points) >= 2:
                     self._path_handle = self._server.scene.add_line_strip(
                         f"{self.name}_path",
                         points,
                         color=(0.0, 1.0, 0.0), # Green path? Or make configurable?
                         line_width=3.0
                     )
             except Exception as e:
                 print(f"Error drawing path: {e}")

        self._update()
        self._draw_nodes.on_update(lambda _: self._update())
        self._draw_labels.on_update(lambda _: self._update())
        self._draw_edges.on_update(lambda _: self._update())
        self._node_scale.on_update(lambda _: self._update())
        self._edge_scale.on_update(lambda _: self._update())
        self._draw_bboxes.on_update(lambda _: self._update())
        self._draw_bbox_labels.on_update(lambda _: self._update())

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
        draw_bbox_labels = self._draw_nodes.value and self._draw_bboxes.value and self._draw_bbox_labels.value
        labels_drawn = len(self._label_handles) > 0
        bbox_labels_drawn = len(self._bbox_label_handles) > 0

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
        if self._bbox_lines_handle:
             self._bbox_lines_handle.visible = draw_bboxes
        if self._bbox_hitbox_handle:
             self._bbox_hitbox_handle.visible = draw_bboxes
        
        # Toggle BBox Labels visibility
        if not draw_bbox_labels and bbox_labels_drawn:
            for label in self._bbox_label_handles:
                label.remove()
            self._bbox_label_handles = []
        
        if draw_bbox_labels and not bbox_labels_drawn:
            self._create_bbox_labels()

        self._parent_callback()

    def _update_bboxes(self, G, layer):
        # Clear existing
        if self._bbox_lines_handle:
            self._bbox_lines_handle.remove()
            self._bbox_lines_handle = None
            
        if self._bbox_hitbox_handle:
            self._bbox_hitbox_handle.remove()
            self._bbox_hitbox_handle = None
        
        # Clear bbox labels
        for label in self._bbox_label_handles:
            label.remove()
        self._bbox_label_handles = []
            
        self._bbox_id_map = []
        
        # Arrays for batching
        all_segments = []
        all_colors = []
        
        hitbox_positions = []
        hitbox_wxyzs = []
        hitbox_scales = []
        
        # Standard unit cube wireframe edges (0..1)
        # However, we get absolute corners from C++. 
        # So we just accumulate segments.
        
        # spark_dsg bitwise corner ordering edges
        edges_bitwise = [
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7),
            (6, 7)
        ]

        for node in layer.nodes:
            if hasattr(node.attributes, "bounding_box"):
                bbox = node.attributes.bounding_box
                if not bbox.is_valid():
                    continue

                try:
                    # --- Wireframe Data ---
                    c_list = bbox.corners()
                    corners = np.array(c_list) 
                    
                    if corners.shape == (8, 3):
                        # Add segments
                        for start, end in edges_bitwise:
                            all_segments.append(corners[start])
                            all_segments.append(corners[end])
                            
                        # Color
                        c = dsg.distinct_150_color(node.attributes.semantic_label).to_float_array() if hasattr(node.attributes, "semantic_label") else (1.0, 0.0, 0.0)
                        c_arr = np.array(c)
                        
                        # We need (12, 2, 3) for the 12 segments, 2 vertices each
                        # Expand c to (1, 1, 3) then tile
                        box_colors = np.tile(c_arr.reshape(1, 1, 3), (12, 2, 1))
                        all_colors.append(box_colors)

                    # --- Hitbox Data ---
                    dim = np.array(bbox.dimensions)
                    center = np.array(bbox.world_P_center)
                    R = np.array(bbox.world_R_center)
                    
                    hitbox_positions.append(center)
                    hitbox_wxyzs.append(vtf.SO3.from_matrix(R).wxyz)
                    hitbox_scales.append(dim)
                    
                    self._bbox_id_map.append(node)

                except Exception as e:
                     print(f"Error preparing bbox batch for {node.id}: {e}")
                     continue

        # create Batched Wireframes
        if all_segments:
            # Reshape to (N, 2, 3) as required by add_line_segments
            points = np.array(all_segments).reshape(-1, 2, 3)
            
            # Colors
            if all_colors:
                colors = np.concatenate(all_colors, axis=0) # (N, 2, 3)
            else:
                colors = (1.0, 0.0, 0.0)
                
            self._bbox_lines_handle = self._server.scene.add_line_segments(
                f"{self.name}_bbox_lines",
                points,
                colors, 
                line_width=2.0
            )

        # Create Batched Hitboxes
            mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
            
            self._bbox_hitbox_handle = self._server.scene.add_batched_meshes_simple(
                f"{self.name}_hitboxes",
                vertices=mesh.vertices,
                faces=mesh.faces,
                batched_positions=np.array(hitbox_positions),
                batched_wxyzs=np.array(hitbox_wxyzs),
                batched_scales=np.array(hitbox_scales),
                opacity=0.0,
            )
            
            self._bbox_hitbox_handle.on_click(self._on_batched_click)


    def _on_batched_click(self, event):
        # event.instance_index contains the index of the clicked instance
        idx = event.instance_index
        if idx is not None and 0 <= idx < len(self._bbox_id_map):
            node = self._bbox_id_map[idx]
            if self._object_manager:
                self._object_manager.handle_click(node)
    
    def _create_bbox_labels(self):
        """Create labels above each bounding box showing object name/ID."""
        # Get labelspace once for efficiency
        labelspace = self._G.get_labelspace(self.key.layer, self.key.partition) if self._G else None
        
        # Prepare all label data first (fast)
        label_data = []
        for node in self._bbox_id_map:
            if not hasattr(node.attributes, "bounding_box"):
                continue
            
            bbox = node.attributes.bounding_box
            if not bbox.is_valid():
                continue
            
            try:
                # Calculate position above the bounding box
                center = np.array(bbox.world_P_center)
                dims = np.array(bbox.dimensions)
                # Position label above the top of the bbox
                label_pos = center.copy()
                label_pos[2] = center[2] + dims[2] / 2.0 + 0.3  # 0.3m above bbox top
                
                # Create label text - show ID with name/category if available
                label_text = str(node.id)
                if hasattr(node.attributes, "name") and node.attributes.name:
                    safe_name = str(node.attributes.name)[:20]  # Truncate long names
                    label_text = f"{node.id} ({safe_name})"
                elif labelspace:
                    category = labelspace.get_node_category(node)
                    if category:
                        label_text = f"{node.id} ({category})"
                
                label_data.append((node.id, label_text, label_pos))
                
            except Exception as e:
                print(f"Error preparing bbox label for {node.id}: {e}")
                continue
        
        # Batch create all labels at once (viser will handle efficiently)
        with self._server.atomic():
            for node_id, label_text, label_pos in label_data:
                try:
                    label_handle = self._server.scene.add_label(
                        f"{self.name}_bbox_label_{node_id}",
                        text=label_text,
                        position=label_pos
                    )
                    self._bbox_label_handles.append(label_handle)
                except Exception as e:
                    print(f"Error creating bbox label for {node_id}: {e}")


    def remove(self):
        if self._nodes:
            self._nodes.remove()
        if self._edges:
            self._edges.remove()

        for x in self._label_handles:
            x.remove()

        self._label_handles = []
        
        for label in self._bbox_label_handles:
            label.remove()
        
        self._bbox_label_handles = []

        self._draw_nodes.remove()
        self._draw_edges.remove()
        self._node_scale.remove()
        self._edge_scale.remove()
        if self._bbox_lines_handle:
            self._bbox_lines_handle.remove()
        if self._bbox_hitbox_handle:
            self._bbox_hitbox_handle.remove()
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
        
        # Add GUI
        self._folder = server.gui.add_folder("Global Mesh")
        with self._folder:
            self._visible = server.gui.add_checkbox("Show Mesh", initial_value=True)
            
        self._visible.on_update(lambda _: setattr(self._mesh_handle, "visible", self._visible.value))

    def remove(self):
        """Remove mesh elements from the visualizer."""
        self._mesh_handle.remove()
        self._folder.remove()


class ViserRenderer:
    """Rendering interface to Viser client."""

    def __init__(self, ip="localhost", port=8080, clear_at_exit=True, image_root=None, image_folder_prefix=None):
        self._server = viser.ViserServer(host=ip, port=int(port))
        self._clear_at_exit = clear_at_exit
        self._mesh_handle = None
        self._graph_handle = None
        self._image_root = image_root
        self._image_folder_prefix = image_folder_prefix
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
        self._object_manager = ObjectManager(self._server, G, self._image_root, self._image_folder_prefix)
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
