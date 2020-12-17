import os


def sprint(s, silent=False):
    """
    A silenceable wrapper around print.
    @param {str} s Text to print
    @param {bool} silent If true, nothing will print. If false, text will be printed.
    """
    if not silent:
        print(s)


def list_paths(file_dir, valid_extensions):
    """
    Return a list of paths to files in a directory with valid extensions
    @param {str} file_dir The directory to find files in
    @param {list} valid_extensions All valid extensions, such as ".tif"
    @return {list} Path to all files with valid extensions in the file directory
    """
    return [os.path.join(file_dir, file) for file in os.listdir(file_dir) if any(file.endswith(ext) for ext in valid_extensions)]


def add_suffix(file_path, suffix):
    """
    Insert a suffix into the file name of a path.
    @param {str} file_path A relative or absolute path to the file, including the file name
    @param {str} suffix A set of characters to append to the file name as a suffix
    @return {str} A relative or absolute path to the file with the suffix appended to the file name
    """
    path, basename = os.path.split(file_path)
    name, ext = os.path.splitext(basename)
    name_suffix = f"{name}{suffix}{ext}"

    full_path = os.path.join(path, name_suffix)
    return full_path
