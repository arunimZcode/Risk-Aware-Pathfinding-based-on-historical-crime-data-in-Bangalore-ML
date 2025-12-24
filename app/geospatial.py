import os
import osmnx as ox
import networkx as nx
import numpy as np
from typing import Dict, Any

# ABSOLUTE IMPORTS to prevent ModuleNotFoundError on reloads
from app.risk_analysis import get_risk_penalty, RISK_MULTIPLIER

# --- CONFIGURATION ---
CITY_NAME = "Bangalore, Karnataka, India"
GRAPHML_PATH = "data/bangalore_graph.graphml"
PROJ_GRAPHML_PATH = "data/bangalore_graph_proj.graphml"

# Cache objects
G_PROJ = None
G_BASE = None 

def load_graph():
    global G_PROJ, G_BASE
    if G_PROJ is not None:
        return G_PROJ

    os.makedirs("data", exist_ok=True)
    
    # Check for cached projected graph first
    if os.path.exists(PROJ_GRAPHML_PATH):
        G_PROJ = ox.load_graphml(PROJ_GRAPHML_PATH)
        if os.path.exists(GRAPHML_PATH):
            G_BASE = ox.load_graphml(GRAPHML_PATH)
        else:
            G_BASE = ox.project_graph(G_PROJ, to_crs="epsg:4326")
        return G_PROJ

    # Otherwise load/download base and project
    if os.path.exists(GRAPHML_PATH):
        G_BASE = ox.load_graphml(GRAPHML_PATH)
    else:
        G_BASE = ox.graph_from_place(CITY_NAME, network_type="drive")
        ox.save_graphml(G_BASE, filepath=GRAPHML_PATH)
    
    if G_BASE.graph.get("crs") != "epsg:4326":
        G_BASE = ox.project_graph(G_BASE, to_crs="epsg:4326")
        
    G_PROJ = ox.project_graph(G_BASE)
    ox.save_graphml(G_PROJ, filepath=PROJ_GRAPHML_PATH)
    return G_PROJ

def get_point_risk(lat: float, lon: float, hour: int, day: str) -> Dict[str, Any]:
    """
    Returns the numerical risk score from the model with a -0.5 adjustment
    to prevent over-sensitizing the risk level in the UI.
    """
    try:
        raw_score = get_risk_penalty(lat, lon, hour, day)
        
        # Apply the -0.5 adjustment for display purposes
        # Clip at 0.0 to ensure we don't return negative risk
        display_score = max(0.0, float(raw_score) - 0.5)
        
        # Categorical level based on adjusted score
        if display_score > 0.8:
            level = "Very High"
        elif display_score > 0.6:
            level = "High"
        elif display_score > 0.4:
            level = "Medium"
        elif display_score > 0.2:
            level = "Low"
        else:
            level = "Very Low"

        return {
            "status": "success",
            "risk_score": round(display_score, 4),
            "level": level,
            "raw_model_score": round(float(raw_score), 4) # Included for debugging if needed
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def _apply_dynamic_weights(G, hour: int, day: str):
    """Calculates risk scores once per node then applies to edges."""
    node_risks = {}
    for node, data in G.nodes(data=True):
        lat = data.get('lat') or data.get('y')
        lon = data.get('lon') or data.get('x')
        # We use raw scores for routing to maintain pathfinding accuracy
        node_risks[node] = get_risk_penalty(lat, lon, hour, day)

    for u, v, k, data in G.edges(keys=True, data=True):
        risk_score = node_risks.get(u, 0.1)
        length_m = data.get('length', 0)
        data['risk_score'] = risk_score
        # Formula: Length * (1 + risk_penalty)
        data['weight'] = length_m * (1 + (risk_score * (RISK_MULTIPLIER / 100)))

def find_safe_route(orig_lat: float, orig_lon: float, dest_lat: float, dest_lon: float, hour: int, day: str):
    base_g = load_graph()
    G = base_g.copy()
    _apply_dynamic_weights(G, hour, day)
    
    try:
        orig_node = ox.nearest_nodes(G_BASE, orig_lon, orig_lat)
        dest_node = ox.nearest_nodes(G_BASE, dest_lon, dest_lat)
        
        route_nodes = nx.shortest_path(G, orig_node, dest_node, weight='weight')
        
        route_coords = []
        total_dist = 0
        total_risk = 0
        
        for i in range(len(route_nodes)):
            node_id = route_nodes[i]
            base_node = G_BASE.nodes[node_id]
            route_coords.append((float(base_node['y']), float(base_node['x'])))
            
            if i < len(route_nodes) - 1:
                u, v = route_nodes[i], route_nodes[i+1]
                edge_data = G.get_edge_data(u, v)
                data = list(edge_data.values())[0] if isinstance(edge_data, dict) else edge_data
                total_dist += data.get('length', 0)
                total_risk += data.get('risk_score', 0)

        # Apply same -0.5 adjustment to the total penalty shown in route results
        avg_risk = total_risk / len(route_nodes)
        adjusted_penalty = max(0.0, avg_risk - 0.5)

        return {
            "status": "success",
            "route": route_coords,
            "total_distance_km": round(total_dist / 1000.0, 2),
            "total_penalty": round(adjusted_penalty, 4)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}