from lattice_image import LatticeImage


name = "Testimage"
N = 5
M = 61

image = LatticeImage(name, M, N)

jpeg_path = "code/static files/test_lattice_image.png"
image.load_from_jpeg(jpeg_path)