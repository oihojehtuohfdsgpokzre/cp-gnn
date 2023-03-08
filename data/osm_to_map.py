import io
import os
import xml.etree.ElementTree as ET
import math
import json

'''
We want to extract a graph of waypoints and connexions.
We also want to value connections between waypoints

Information are extracted from OpenStreetMap data:
We need to find roads with at least 2 connected waypoints in the area.

Note: Generated graph will be compressed before planning: replace series of waypoints by single connection
'''

def waypoint_in_area(lat, lon, area):
    
    if lat < area['min_lat']:
        return False
    if lat > area['max_lat']:
        return False
    if lon < area['min_lon']:
        return False
    if lon > area['max_lon']:
        return False        
    return True

def NodeDict(wp, lat, lon):
    res = dict()
    res['id'] = wp
    res['lat'] = lat
    res['lon'] = lon
    return res
    
def EdgeDict(src, dst, lenght):
    res = dict()
    res['src'] = src
    res['dst'] = dst
    res['distance'] = lenght
    return res    

def waypointDistance(lat1, lon1, lat2, lon2):
    R = 6373.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

extracted_road_types = dict()

# Get every node ID from a given elem if it is a way
def extractRoadWaypoints(root, road_types):
    is_road = False
    nodes = []
    road_type = ''
    global extracted_road_types # debug only: get the list of extracted road types
    
    for elem in root:
        if elem.tag == 'tag' and elem.attrib['k'] == 'highway':
            road_type = elem.attrib['v']

            if road_type in road_types:
                is_road = True

    # Extract nodes if the elem is a way
    if is_road == True:
        for elem in root:
            if elem.tag == 'nd':    
                node_id = elem.attrib['ref']
                nodes.append(node_id)
                    
        if road_type not in extracted_road_types.keys():
            extracted_road_types[road_type] = 1
        else:
            extracted_road_types[road_type] += 1                      
    return nodes

def main():      
    print('converting osm to map files')
    
    for folder, dirs, files in os.walk(os.path.join('.', 'osm'), topdown=True):
        for name in files:
            if 'config' in name:
            
                # Load configuration
                conf_file = open(os.path.join(folder, name), 'r')
                config = json.load(conf_file)
                conf_file.close()    
                
                # Load openstreetmap file
                openstreetmap_file = os.path.join('.', 'osm', config['map_file'])
                print('converting', openstreetmap_file)
                tree = ET.parse(openstreetmap_file)
                root = tree.getroot()
                
                print('LOADED', root.tag)
                
                ''' Extract all nodes in the area, even if not used on ways, and store lat/long '''
                node_list = dict() # Use dict to speed information retrieval
                for child in root:
                    if child.tag == 'node':
                        p_lat = float(child.attrib['lat'])
                        p_lon = float(child.attrib['lon'])
                        # Check if node is inside mission area
                        if waypoint_in_area(p_lat, p_lon, config['area']):
                            # node id is expected to be unique
                            node_list[child.attrib['id']] = (p_lat, p_lon)
                
                print('number of nodes extracted in the area : ', len(node_list.keys()))
                
                # Find node common between ways, which are intersections
                way_in_area = dict()
                
                graph_nodes = dict()
                graph_edges = []
                
                for child in root:
                    if child.tag == 'way':
                        waypoints = extractRoadWaypoints(child, config['road_types'])
                        if len(waypoints) == 0:
                            continue
                        
                        way_id = child.attrib['id']
                        way_in_area[way_id] = waypoints


                        # Extract every segment of road, with start and end of each segment in the mission area
                        # 2 pointers are kept: previous node, and current node.
                        # After this loop, a dictionnary of nodes, and a list of edge, are built.
                        wp_number = len(waypoints)
                        wp_id_prev = waypoints[0]
                        ind = 1
                        while(ind < wp_number):
                            wp = waypoints[ind]
                            
                            if wp in node_list.keys() and wp_id_prev in node_list.keys():
                                (lat_prev, lon_prev) = node_list[wp_id_prev]
                                graph_nodes[wp_id_prev] = (lat_prev, lon_prev)
                                
                                (lat, lon) = node_list[wp]
                                graph_nodes[wp] = (lat, lon)

                                dist = waypointDistance(lat, lon, lat_prev, lon_prev)
                                graph_edges.append(EdgeDict(wp_id_prev, wp, dist))

                            wp_id_prev = wp
                            ind += 1

                    
                global extracted_road_types # debug only: get the list of extracted road types    

                print('extracted road types : ', extracted_road_types)
                
                ''' JSON Graph file generation '''
                problem = dict()
                problem['graph_nodes'] = graph_nodes
                problem['graph_edges'] = graph_edges
                
                print('Number of nodes extracted: ', len(graph_nodes))
                
                dest_name = os.path.join('.', 'maps', config['graph_file'])
                dest_file = open(dest_name, 'w')
                
                json.dump(problem, dest_file, indent=4)
                dest_file.close()
    
main()

