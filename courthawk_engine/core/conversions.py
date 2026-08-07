"""Conversions between pixels and meters."""


def convert_pixel_distance_to_meters(
        pixel_distance: float, 
        reference_meters: float, 
        reference_pixels: int
) -> float:
    return pixel_distance * (reference_meters / reference_pixels)


def convert_meters_to_pixel_distance(
        meters: float,
        reference_meters: float,
        reference_pixles: int
) -> float:
    return meters * (reference_pixles / reference_meters)