from __future__ import print_function

# Mapping related imports --------------------------------------------------------------------------
import smopy
import geopy
from geopy.distance import VincentyDistance

# Math/Image imports -------------------------------------------------------------------------------
import math
import numpy as np
import cv2
from docutils.nodes import row

def display_image(win_name, image):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, image)
    cv2.waitKey(0)

class MapSection():
    '''
    Container for smopy maps and some other information that's useful to have with them
    '''
    def __init__(self, min_lat, max_lat, min_lon, max_lon, z=-1):
        '''
        Constructor for the class
        @param min_lat: The bounding minimum latitude
        @param max_lat: The bounding maximum latitude
        @param min_lon: The bounding minimum longitude
        @param max_lon: The bounding maximum longitude
        @param tile_x: The x location of this tile in the larger map
        @param tile_y: The y location of this tile in the larger map
        '''
        # Save the bounding values
        self._min_lat = min_lat
        self._max_lat = max_lat
        
        self._min_lon = min_lon
        self._max_lon = max_lon
        
        # Establish attributes for the u v bounds
        self._u_min = None
        self._v_min = None
        self._u_max = None
        self._v_max = None
        
        # Save the tile coordinates
        self._tile_x = None
        self._tile_y = None
        
        # Create the smopy map
        self._map = None
        self._zoom = None
        if z == -1:
            self._map = smopy.Map((self._min_lat, self._min_lon, self._max_lat, self._max_lon))
            self._zoom = self._map.z
        else:
            self._zoom = z
            self._map = smopy.Map((self._min_lat, self._min_lon, self._max_lat, self._max_lon), z=z)
        self.populate_limits_uv()

    def value_in_tile(self, lat, lon):
        '''
        Determines whether a passed point falls within this tile
        @param lat: latitude of the point
        @param lon: longitude of the point
        @return: True if point is in region covered by tile, False if not
        '''
        
        # Check if the latitude is in the region
        if (lat >= self._min_lat) and (lat <= self._max_lat):
            # Check if the longitude is in the region
                if (lon >= self._min_lon) and (lon <= self._max_lon):
                    return True
        
        # If the if statement didn't return, return False
        return False
    
    # TODO: figure out if I can just do this in the final image with some magic
    def to_pixels_uv(self, lat, lon):
        '''
        Fetches the corresponding pixel coordinates of a given lat and lon within the map tile
        @param lat: The latitude of the point
        @param lon: The longitude of the point
        @return: Returns a tuple (u,v) of the pixel location
        '''
        
        # Double check the value is actually in the tile
        if not self.value_in_tile(lat, lon):
            return False
        
        # Fetch the u, v values of the lat lon
        u, v = self._map.to_pixels(lat, lon)
        return (u, v)
    
    def populate_limits_uv(self):
        # Get the pixel values of the latitude longitude corners
        corner0 = self._map.to_pixels(self._min_lat, self._min_lon)
        corner1 = self._map.to_pixels(self._max_lat, self._min_lon)
        corner2 = self._map.to_pixels(self._min_lat, self._max_lon)
        corner3 = self._map.to_pixels(self._max_lat, self._max_lon)
        
        # Find the minimum and maximum u, v values
        self._u_min = int(corner0[0])
        self._v_min = int(corner3[1])
        self._u_max = int(corner2[0])
        self._v_max = int(corner0[1])

    
    def get_refined_image(self):
        '''
        Processes the image, translating cropping
        @return: "size_x, size_y" returns the dimensions of the image
        '''
        image = self._map.to_numpy()
        
        # Get the translational and cropping parameters
        origin_point = (self._u_min, self._v_min)
        crop_width = self._u_max - self._u_min
        crop_height = self._v_max - self._v_min
        h_translation = -(self._u_min)
        v_translation = -(self._v_min)
        
        row, col = image.shape[:2]
        
        # Create the translational matrix for use in warpAffine
        trans_mat = np.float32([[1, 0, h_translation],
                                [0, 1, v_translation]])
        
        image = cv2.warpAffine(image, trans_mat, (col, row))
        
        # Crop the image
        image = image[0:crop_height - 1, 0:crop_width - 1]
            
        return image
        

   
class MapStitcher():
    '''
    Retrieves a map that inscribes a region given by the minimum and
    maximum latitudes and longitudes, a higher sq_size value gets a higher
    resolution image
    '''
    
    def __init__(self, min_lat, max_lat, min_lon, max_lon, sq_size=5):
        '''
        Constructor for the class
        @param min_lat: The bounding minimum latitude
        @param max_lat: The bounding maximum latitude
        @param min_lon: The bounding minimum longitude
        @param max_lon: The bounding maximum longitude
        @param sq_size: The square size of how many map tiles to stitch (higher = higher res & longer run time)
        '''
        
        # Save the bounding values
        self._min_lat = min_lat
        self._max_lat = max_lat
        
        self._min_lon = min_lon
        self._max_lon = max_lon
        
        # Save the square size
        self._sq_size = sq_size
        
        # This will contain the final image after all processing
        self._stitched_map_image = None
        
        # This contains all of the map sections
        self._map_sections = []
        
        # Fill everything up
        self.populate_map_sections()
        
        # Create the final image
        self.stitch_sections()
        
    def populate_map_sections(self):
        '''
        Populates "self._map_sections" with all of the map sections in the larger map
        '''
        
        # Calculate the total latitude and longitude range of the whole map
        lat_range = self._max_lat - self._min_lat
        lon_range = self._max_lon - self._min_lon
        
        # Calculate the step size of each tile
        lat_step_size = lat_range / self._sq_size
        lon_step_size = lon_range / self._sq_size
        
        # Now create a list containing all of the divisions of latitude and longitude
        lat_increments = []
        lon_increments = []
        for iter in range(self._sq_size + 1):
            # How much we need to go from the minimum
            lat_step = iter * lat_step_size
            lon_step = iter * lon_step_size
            
            # Add the step to the list
            lat_increments.append(self._min_lat + lat_step)
            lon_increments.append(self._min_lon + lon_step)
        
        
        # Create a list of all the corners of the map_sections
        stitch_corners = []
        for lat_iter in range(self._sq_size):
            for lon_iter in range(self._sq_size):
                corner = (lat_increments[lat_iter], lat_increments[lat_iter + 1],
                          lon_increments[lon_iter], lon_increments[lon_iter + 1])
                stitch_corners.append(corner)
                
        # Create all of the MapSection objects
        is_first_map = True
        zoom_val = -1
        x_size = 0
        y_size = 0
        map_counter = 1
        for corners in stitch_corners:
            print("Fetching map section : [ ", map_counter, " / ", self._sq_size*self._sq_size, "]")
            map_counter += 1
            # Figure out what the right zoom is if it's the first one
            if is_first_map:
                map = MapSection(min_lat=corners[0], max_lat=corners[1],
                                 min_lon=corners[2], max_lon=corners[3], z=-1)
                zoom_val = map._zoom
                is_first_map = False
            # Otherwise get the map with the determined zoom level
            else:
                map = MapSection(min_lat=corners[0], max_lat=corners[1],
                                 min_lon=corners[2], max_lon=corners[3], z=zoom_val)
                if map._zoom != zoom_val:
                    print("Zoom level mismatch the map is going to be wrong.")
            
            self._map_sections.append(map)
    
    def stitch_sections(self):
        is_first_tile = True
        tile_rows = []
        first_tile = self._map_sections[0].get_refined_image()
        tile_height = first_tile.shape[0]
        tile_width = first_tile.shape[1]
        for row_index in range(self._sq_size):
            row = None
            is_first_tile = True
            for tile_index in range(self._sq_size):
                adjusted_index = (row_index * self._sq_size) + tile_index
                cur_tile = self._map_sections[adjusted_index].get_refined_image()
                print("Resizing from dimensions of: (", cur_tile.shape[1], ", ", cur_tile.shape[0], ")",
                      " -->  (", tile_width, ", ", tile_height, ")"  )
                cur_tile = cv2.resize(cur_tile, (tile_width, tile_height))
                if is_first_tile:
                    row = cur_tile
                    is_first_tile = False
                else:
                    row = np.hstack((row, cur_tile))
            tile_rows.append(row)
            
        stacked = None
        is_first_row = True
        for row in reversed(tile_rows):
            if is_first_row:
                stacked = row
                is_first_row = False
            else:
                stacked = np.vstack((stacked, row))
        self._stitched_map_image = stacked
    
    def to_pixels_uv(self, lat, lon):
        '''
        Converts a latitude longitude value to a pixel location
        @param lat: The latitude to convert
        @param lon: The longitude to convert
        @return: u, v : The u and v coordinates of the geo point
        '''
        # Get the total pixels in image
        image_height = self._stitched_map_image.shape[0]
        image_width = self._stitched_map_image.shape[1]
        
        # Calculate the total latitude and longitude range of the whole map
        lat_range = self._max_lat - self._min_lat
        lon_range = self._max_lon - self._min_lon
        
        # Calculate the amount of lat/lon per pixel
        pixel_per_lat = image_height / lat_range
        pixel_per_lon = image_width / lon_range
        
        # Make sure it's in the map
        if lat < self._min_lat or lat > self._max_lat:
            if lon < self._min_lon or lon > self._max_lon:
                return False
        
        # Calculate the lat/lon change from the boundary
        lat_change = self._max_lat - lat
        lon_change = lon- self._min_lon
        
        lat_pixels = lat_change * pixel_per_lat
        lon_pixels = lon_change * pixel_per_lon
        
        lat_pixels = round(lat_pixels)
        lon_pixels = round(lon_pixels)
        return int(round(lon_pixels)), int(round(lat_pixels))
    
    def get_image(self):
        return self._stitched_map_image

if __name__ == '__main__':
    lat = 42.363216
    lon = -71.086175
    origin = (lat, lon)
    bearing = 100
    range_nmi = 2.5
    sweep_angle = 60
  
    ms = MapStitcher(min_lat=42.3204153868, max_lat=42.3987644538, 
                     min_lon=-71.0946224618, max_lon=-71.0223426501, sq_size=3)
    final_image = ms.get_image()
  
    cv2.namedWindow("Output Map", cv2.WINDOW_NORMAL)
    cv2.imshow("Output Map", final_image)
    cv2.waitKey(0)
    
    
    
    
    
    