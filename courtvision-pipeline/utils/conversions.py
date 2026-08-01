def convert_pixel_distance_to_meters(
        pixel_distance, 
        reference_meters, 
        reference_pixels
):
    return pixel_distance * (reference_meters / reference_pixels)

def convert_meters_to_pixel_distance(
        meters,
        reference_meters,
        reference_pixles
):
    return meters * (reference_pixles / reference_meters)