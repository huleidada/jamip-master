from packaging import version

def get_spglib_version():
    """
    Get the version of the spglib package.
    
    Returns:
        str: The version of spglib.
    """
    import spglib
    v = version.parse(spglib._version.version)
    benchmark = version.parse('2.0.0')
    return v >= benchmark