
import pytest
import unittest
from unittest.mock import MagicMock
import numpy as np
import spark_dsg.viser as viser_module
from spark_dsg.viser import ObjectManager

class TestViserRegression(unittest.TestCase):
    def setUp(self):
        self.mock_server = MagicMock()
        self.mock_G = MagicMock()
        self.mock_node = MagicMock()
        self.mock_node.id.value = 1
        self.mock_node.id = 1 # Int ID
        
        # Mock Read-Only Array
        ro_array = np.array([0.0, 0.0, 0.0])
        ro_array.flags.writeable = False # Simulate C++ view
        self.mock_node.attributes.position = ro_array
        
        self.mock_G.get_node.return_value = self.mock_node
        self.mock_G.has_node.return_value = True
        
        self.om = ObjectManager(self.mock_server, self.mock_G)
        self.om._current_node = self.mock_node
        self.om._transform_controls = MagicMock() 
        
        # Mock UI elements needed for _on_coord_change
        self.om._coord_x = MagicMock()
        self.om._coord_x.value = 5.0
        self.om._coord_y = MagicMock()
        self.om._coord_y.value = 5.0
        self.om._coord_z = MagicMock()
        self.om._coord_z.value = 5.0
        self.om._rot_r = MagicMock()
        self.om._rot_r.value = 0.0
        self.om._rot_p = MagicMock()
        self.om._rot_p.value = 0.0
        self.om._rot_y = MagicMock()
        self.om._rot_y.value = 0.0
        
    def test_coord_change_readonly_fix(self):
        """
        Regression Test: Ensures that modifying read-only position attributes
        (e.g. from C++ bindings) via UI inputs creates a new array instead of 
        modifying in-place, avoiding ValueError.
        Also validates that no NameError occurs (variable name regression check).
        """
        try:
            self.om._on_coord_change(None)
        except ValueError as e:
            self.fail(f"_on_coord_change raised ValueError: {e}")
        except NameError as e:
             self.fail(f"_on_coord_change raised NameError: {e}")
            
        # Verify new array was assigned
        self.assertTrue(np.array_equal(self.om._current_node.attributes.position, np.array([5.0, 5.0, 5.0])))
        
    def test_gizmo_update_readonly_fix(self):
        """
        Regression Test: Ensures that dragging the gizmo works with read-only attributes.
        """
        # Setup Gizmo controls mock
        self.om._transform_controls.position = np.array([2.0, 3.0, 4.0])
        
        try:
            self.om._on_gizmo_update(None)
        except ValueError as e:
             self.fail(f"_on_gizmo_update raised ValueError: {e}")
             
        # Verify assignment
        self.assertTrue(np.array_equal(self.om._current_node.attributes.position, np.array([2.0, 3.0, 4.0])))

    def test_apply_callback(self):
        """
        Validates that clicking "Apply Transform" triggers the connected callback with client.
        """
        callback = MagicMock()
        self.om.on_transform_apply = callback
        
        event = MagicMock()
        event.client = MagicMock()
        self.om._on_apply_click(event)
        
        callback.assert_called()
        args = callback.call_args
        self.assertEqual(args[0][0], 1) # Node ID
        self.assertEqual(args[0][2], event.client) # Client passed
        
    def test_graph_handle_sync(self):
        """
        Validates that gizmo/coord updates trigger GraphHandle visual updates.
        """
        mock_gh = MagicMock()
        self.om._graph_handle = mock_gh
        
        # Test Gizmo Update
        self.om._transform_controls.position = np.array([10.0, 10.0, 10.0])
        self.om._on_gizmo_update(None)
        
        mock_gh.update_node_visuals.assert_called_with(1)
        
        # Test Coord Update
        mock_gh.reset_mock()
        self.om._on_coord_change(None)
        mock_gh.update_node_visuals.assert_called_with(1)
        
if __name__ == '__main__':
    unittest.main()
