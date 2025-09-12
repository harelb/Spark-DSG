#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/edge_attributes.h>
#include <spark_dsg/mesh.h>

#include <iostream>
#include <string>

// Use the spark_dsg namespace for convenience
using namespace spark_dsg;

// Define the layer IDs for clarity and easy modification
const LayerId OLD_LAYER_ID = 20;
const LayerKey NEW_LAYER_KEY = {3, 1}; // Standard key for MESH_PLACES
const std::string NEW_LAYER_NAME = "MESH_PLACES";


// A helper function to print usage instructions
void print_usage() {
    std::cerr << "Usage: dsg_migrator <input_dsg_filepath> <output_dsg_filepath>" << std::endl;
}

int main(int argc, char* argv[]) {
    // --- 1. Argument Parsing ---
    if (argc != 3) {
        print_usage();
        return EXIT_FAILURE;
    }

    const std::string input_filepath = argv[1];
    const std::string output_filepath = argv[2];

    std::cout << "Loading graph from: " << input_filepath << std::endl;

    // --- 2. Load the Original Graph ---
    DynamicSceneGraph::Ptr original_graph = DynamicSceneGraph::load(input_filepath);
    if (!original_graph) {
        std::cerr << "Failed to load graph from: " << input_filepath << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Graph loaded successfully." << std::endl;

    // --- 3. Create a New Graph for the Corrected Data ---
    DynamicSceneGraph new_graph;

    // --- 4. The Migration Logic ---
    std::cout << "Starting migration..." << std::endl;
    std::cout << "Remapping Layer " << OLD_LAYER_ID << " -> (Layer: " << NEW_LAYER_KEY.layer 
              << ", Partition: " << NEW_LAYER_KEY.partition << ")" << std::endl;

    for (const auto& id_layer_pair : original_graph->layers()) {
        // FIX: Use the arrow operator '->' to access members via the smart pointer.
        for (const auto& id_node_pair : id_layer_pair.second->nodes()) {
            const auto& node = id_node_pair.second;
            auto new_attrs = node->attributes().clone();

            if (node->layer.layer == OLD_LAYER_ID) {
                new_graph.emplaceNode(NEW_LAYER_KEY, node->id, std::move(new_attrs));
            } else {
                new_graph.emplaceNode(node->layer, node->id, std::move(new_attrs));
            }
        }
    }
    std::cout << "Nodes migrated: " << new_graph.numNodes() << std::endl;

    for (const auto& id_layer_pair : original_graph->layers()) {
        // FIX: Use the arrow operator '->' to access members via the smart pointer.
        for (const auto& id_edge_pair : id_layer_pair.second->edges()) {
            const auto& edge = id_edge_pair.second;
            new_graph.insertEdge(edge.source, edge.target, edge.info->clone());
        }
    }
    for (const auto& id_edge_pair : original_graph->interlayer_edges()) {
        const auto& edge = id_edge_pair.second;
        new_graph.insertEdge(edge.source, edge.target, edge.info->clone());
    }
    std::cout << "Edges migrated: " << new_graph.numEdges() << std::endl;

    if (original_graph->hasMesh()) {
        new_graph.setMesh(std::make_shared<Mesh>(*original_graph->mesh()));
        std::cout << "Mesh data migrated." << std::endl;
    }

    new_graph.addLayer(NEW_LAYER_KEY.layer, NEW_LAYER_KEY.partition, NEW_LAYER_NAME);
    std::cout << "Assigned new layer name for MESH_PLACES." << std::endl;

    // --- 5. Save the New Graph ---
    std::cout << "Saving corrected graph to: " << output_filepath << std::endl;
    new_graph.save(output_filepath);

    std::cout << "Migration complete." << std::endl;

    return EXIT_SUCCESS;
}
