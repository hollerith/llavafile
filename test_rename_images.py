import os
import json
from unittest.mock import patch, MagicMock
from rename_images import PROMPT, load_known_faces, get_image_files, generate_new_name, rename_images

def test_load_known_faces():
    # Mock the file reading and pickle loading
    with patch('rename_images.pickle.load', return_value={'encodings': [], 'names': []}) as mock_pickle_load:
        with patch('rename_images.open', return_value=MagicMock()) as mock_open:
            encodings, names = load_known_faces('known_faces.dat')
            assert encodings == []
            assert names == []
            mock_open.assert_called_once_with('known_faces.dat', 'rb')
            mock_pickle_load.assert_called_once()

def test_get_image_files():
    # Mock the glob.glob function to return a list of image paths
    with patch('rename_images.glob.glob', return_value=['image1.jpg', 'image2.jpg']) as mock_glob:
        image_files = get_image_files('test_folder', '*.jpg')
        assert image_files == ['image1.jpg', 'image2.jpg']
        mock_glob.assert_called_once_with('test_folder/*.jpg')
