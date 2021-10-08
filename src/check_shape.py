# ==============================================================================
# ShapeError
# ==============================================================================

class ShapeError(Exception):
    pass


# ==============================================================================
# Shape checker
# ==============================================================================


def check_shape(arrays, shapes, shape_dict=None, keep_dict=False):
    """
    Checks the shape(s) or **arrays** against **shapes**. There's two
    types of usage. Either
    
        (1) **arrays** is an array and shapes is a tuple
        (2) **arrays** is a list of arrays and shapes a list of tuples
        
    In case (1), the array is checked against the shape. It is assumed
    that the array has a .shape property. It is also assumed that the
    **shapes** argument is a (possibly mixed) tuple of int and str.
    
    In case (2), each array is checkeed against the corresponding shape.
    It is assumed that each array has a .shape property. It is also
    assumed that **shape** is a (possibly mixed) tuple of int and str.
    
    Arguments:
        arrays     : (1) array; or (2) [array], array(s) to check
        shapes     : (1) tuple; or (2) [tuple], shape(s) to check
        shape_dict : dict {str : int}, axis name to axis size dict
        keep_dict  : bool, whether to return a shape dict
    """
    
    # Use case (1): single array and single shape
    if (type(arrays) != list) and (type(shapes) != list):
        return _check_shape(array=arrays,
                            shape=shapes,
                            shape_dict=shape_dict)
    
    # Use case (2): mutliple arrays and shapes
    if (type(arrays) == list) and (type(shapes) == list):
        
        # If shape_dict was not provided, start a new shape_dict
        shape_dict = {} if shape_dict is None else shape_dict
        
        # Check number of arrays and number of shapes passed are the same
        if len(arrays) != len(shapes):
            raise ShapeError(f"Got number of tensors/arrays {len(arrays)}, "
                             f"and number of shapes {len(shapes)}.")
            
        # Iterate over arrays and shapes
        for argnum, (array, shape) in enumerate(zip(arrays, shapes)):
            
            if type(shape) == str:
                raise ShapeError(f"Shape {shape} should have type tuple, "
                                 f"got str instead.")
            
            # Convert shape to tuple
            try:
                shape = tuple(shape)
                
            except:
                raise ShapeError(f"Shape {shape} of type {type(shape)}, "
                                 f"cannot be converted to tuple.")
                
            # Check array/shape have same dimensions
            assert len(array.shape) == len(shape)
            
            # Do shape checking
            array, shape_dict = _check_shape(array,
                                             shape,
                                             shape_dict=shape_dict,
                                             argnum=argnum)
            
        # If keep_dict, return both the arrays and the shape_dict
        if keep_dict:
            return arrays, shape_dict
        
        # Otherwise return arrays only
        else:
            return arrays
    
    # No other use cases supported
    else:
        raise ShapeError(f"Invalid combination of arrays and shapes passed.")

        

def _check_shape(array, shape, shape_dict=None, argnum=None):
    
    array_shape = array.shape
    check_string_names = shape_dict is not None
    shape_dict = {} if shape_dict is None else shape_dict
    
    # Check if array shape and shape have same length (i.e. are comparable)
    if len(array_shape) != len(shape):
        raise ShapeError(f"Tensor/Array shape {array_shape}, "
                         f"check shape {shape}")
    
    # Check if shapes are compatible
    for s1, s2 in zip(array.shape, shape):
        
        assert type(s2) in [str, int]
        
        # Try to convert s2 to int
        try:
            
            # If s2 == '-1', any shape passes test
            if int(s2) == -1:
                continue

            elif s1 != int(s2):
                raise ShapeError(f"Tensor/Array shape {array_shape}, "
                                 f"check shape {shape}")
                
        # If s2 string found, try to match against dict
        except ValueError:
            
            # If s2 string not in shape dict, update shape dict
            if not (s2 in shape_dict):
                shape_dict[s2] = s1
                
            # Elif string in shape dict, but shapes incompatible, raise error
            elif shape_dict[s2] != s1:
                raise ShapeError(f"Tensor/Array at argument position {argnum} "
                                 f"had shape with {s2} of size {s1}, "
                                 f"expected axis size {shape_dict[s2]}.")
            
    if check_string_names:
        return array, shape_dict
    
    else:
        return array
