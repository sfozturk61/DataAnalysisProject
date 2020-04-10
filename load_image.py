from lattice_image import LatticeImage


name = "Testimage"
N = 5
M = 61

lattice_image = LatticeImage(
    name=name,
    M=M,
    N=N
)

jpeg_path = "code/static files/test_lattice_image.png"
lattice_image.load_from_jpeg(jpeg_path)

lattice_image.structure_image()