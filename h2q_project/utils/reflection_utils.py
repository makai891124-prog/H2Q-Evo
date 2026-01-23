from h2q_project.core.geometry import GeometryReflection

def reflect_geometry(geometry_object, line_segment):
    # Use the GeometryReflection class for reflection
    reflection_tool = GeometryReflection()
    
    if hasattr(geometry_object, 'x') and hasattr(geometry_object, 'y'):
        return reflection_tool.reflect_point_across_line(geometry_object, line_segment)
    else:
        return None


def describe_object(geometry_object):
    reflection_tool = GeometryReflection()
    return reflection_tool.describe_geometry(geometry_object)