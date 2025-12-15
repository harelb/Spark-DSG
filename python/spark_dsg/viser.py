"""Entry point for visualizing scene graph."""

import enum
import functools
import itertools
from dataclasses import dataclass

import numpy as np
import trimesh
import viser
import viser.transforms as vtf
import json
import io
import logging
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
            self._show_3d_image = server.gui.add_checkbox("Show 3D Image", initial_value=True)
            self._maximize_2d = server.gui.add_checkbox("Maximize 2D Image", initial_value=False)
            
            self._image_handle_2d = None
            self._image_handle_3d = None
            self._image_label_3d = None
            self._mesh_handle = None

        self._current_node = None
        self._node_map = {} # label -> node_id
        
        # Populate dropdown
        self._update_dropdown()
        
        # Event callbacks
        self._object_dropdown.on_update(self._on_object_select)
        self._jump_button.on_click(self._on_jump)
        self._toggle_mesh.on_update(self._on_view_update)
        self._toggle_bbox_2d.on_update(self._on_view_update)
        self._show_3d_image.on_update(self._on_view_update)
        self._maximize_2d.on_update(self._on_view_update)

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

    def _on_object_select(self, event):
        val = self._object_dropdown.value
        if val == "None":
            self._current_node = None
            self._clear_visuals()
            return

        node_id = self._node_map[val]
        self._current_node = self._G.get_node(node_id)
        self._update_selection()

    def _on_jump(self, event):
        if self._current_node and event.client:
            pos = self._current_node.attributes.position
            event.client.camera.look_at = pos
            # Zoom out a bit
            event.client.camera.position = pos + np.array([-3.0, -3.0, 3.0])

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
        # Find image
        image_path, meta_path = self._find_image_paths(self._current_node)
        if not image_path:
            return
            
        try:
            # Load Image
            pil_image = Image.open(image_path)
            
            # Draw BBox if requested (for 2D only)
            image_for_2d = pil_image.copy()
            if self._toggle_bbox_2d.value and meta_path and meta_path.exists():
                self._draw_2d_bbox(image_for_2d, meta_path)

            # Convert to numpy for viser
            img_array_2d = np.array(image_for_2d)
            img_array_raw = np.array(pil_image)
            
            # Retrieve Object Name
            obj_name = self._get_object_label(self._current_node)

            # 2D Side Panel Image
            with self._image_container:
                self._image_handle_2d = self._server.gui.add_image(
                    img_array_2d,
                    label=f"{obj_name} ({self._current_node.id})",
                    format="jpeg"
                )
            
            # 3D Scene Image
            if self._show_3d_image.value:
                # Position above the object?
                pos = self._current_node.attributes.position
                # Default size?
                # Using add_image in scene
                # Need to determine dimensions or scale. 
                # Let's pick a reasonable width like 1.0 or 2.0 meters?
                width = 2.0
                # If "Maximize" is checked, maybe make it huge in 3D?
                if self._maximize_2d.value:
                    width = 5.0
                
                height = width * (img_array_raw.shape[0] / img_array_raw.shape[1])
                
                # Orientation: Face up (Vertical)
                # Previous +pi/2 resulted in upside down.
                # Let's try -pi/2.
                wxyz = vtf.SO3.from_x_radians(-np.pi/2).wxyz
                
                img_pos = pos + np.array([0, 0, 1.0])
                
                self._image_handle_3d = self._server.scene.add_image(
                    f"/image_3d_{self._current_node.id}",
                    image=img_array_raw,
                    render_width=width,
                    render_height=height,
                    position=img_pos, # 1m above center
                    wxyz=wxyz,
                    format="jpeg"
                )
                
                # Add 3D Label above image
                # Image is centered at img_pos with height 'height'.
                # Top of image is img_pos.z + height/2 (since vertical)
                label_pos = img_pos + np.array([0, 0, height/2 + 0.2])
                
                self._image_label_3d = self._server.scene.add_label(
                    f"/image_label_{self._current_node.id}",
                    text=f"{obj_name} ({self._current_node.id})",
                    position=label_pos
                )
            
        except Exception as e:
            print(f"Error loading image for {self._current_node.id}: {e}")

    def _find_image_paths(self, node):
        # logging less spam
        # Try to find image folder from attributes
        if not hasattr(node.attributes, "image_folder"):
            return None, None
            
        folder = node.attributes.image_folder
        if not folder:
            return None, None
            
        folder_path = Path(folder)
        if self._image_root:
            # If folder is absolute, image_root might be ignored or used to re-root?
            # Usually strict append if relative.
            if not folder_path.is_absolute():
                folder_path = self._image_root / folder_path
            
        # print(f"  Checking folder path: {folder_path}")
        if not folder_path.exists():
            # Fallback for "temp" paths: .../images/temp/O_XX -> .../images/O_{node_id}
            if "temp" in folder_path.parts:
                try:
                    # Find 'temp' index and go up one level
                    # Assuming structure .../images/temp/O_XX
                    # We want .../images/O_{node.id.category_id}
                    # If we perform path manipulation:
                    new_path = Path(str(folder_path).replace("/temp/", "/"))
                    # But we also need to change the folder name from O_TRACKID to O_NODEID
                    # So proper way is to find the 'images' dir (parent of temp)
                    # This is heuristic based on user input
                    parent = folder_path.parent.parent
                    fallback_node = parent / f"O_{node.id.category_id}"
                    
                    # print(f"  Temp path detected. Trying fallback: {fallback_node}")
                    if fallback_node.exists():
                        folder_path = fallback_node
                except Exception as e:
                    pass # print(f"  Error in path fallback: {e}")

        
        if not folder_path.exists():
            # Fallback: maybe just image_root / O_<id>?
            if self._image_root:
                fallback = self._image_root / f"O_{node.id.category_id}"
                # print(f"  Folder not found. Checking fallback root: {fallback}")
                if fallback.exists():
                    folder_path = fallback
                # Try finding folder by name O_ID
                else: 
                     # Search inside root?
                     pass
        
        if not folder_path.exists():
             # print("  Folder does not exist")
             return None, None

        # Find first RGB image
        # Pattern: frame_*_rgb.jpg
        images = list(folder_path.glob("*_rgb.jpg")) + list(folder_path.glob("*_rgb.png"))
        # print(f"  Found images: {len(images)} in {folder_path}")
        if not images:
            return None, None
            
        # Pick the first one or middle one? User didn't specify. First is fine.
        image_file = images[0]

        
        # Find corresponding meta
        # Pattern: frame_{timestamp}_rgb.jpg -> frame_{timestamp}_meta.json
        # Assuming format frame_TIMESTAMP_suffix
        parts = image_file.name.split('_')
        # frame, timestamp, rgb.jpg
        if len(parts) >= 3:
            timestamp = parts[1]
            meta_file = folder_path / f"frame_{timestamp}_meta.json"
        else:
            meta_file = None
            
        return image_file, meta_file

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
