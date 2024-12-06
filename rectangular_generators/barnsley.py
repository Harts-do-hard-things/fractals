from manim import *
from ifsread import ifs

import random
import math

# config.frame_height = 16
# config.frame_width = 9
# config.pixel_width = 1080
# config.pixel_height = 1920
config.frame_rate = 60
# config.renderer = "opengl"
# config.background_color = '#455D3E'


def divide_dots(dots, dots_before):
    start = math.log(len(dots), dots_before)
    dots_groups = [dots[i] for i in range(dots_before)]
    for i in range(1, int(start)):
        for j in range(1, dots_before):
            dots_groups.append(VGroup(*dots[dots_before ** i * j:dots_before ** (i) * (j + 1)]))
    dots_groups.append(VGroup(*dots[dots_before ** (int(start)):]))
    return dots_groups



class Fern(Scene):
    def construct(self):
        colors = [RED, BLUE, YELLOW, PURPLE]
        Dot.set_default(radius=0.03, color=WHITE)
        fern = ifs.IFS_coral()
        fern.iterate(2**11)
        print(fern.xlim)
        ax = Axes(x_range=(-2.7, 2.7), y_range=(0, 10), y_length=6).to_edge(DOWN)
        def ifs_to_dots(ifs, ax) -> list[Dot]:
            return [Dot(ax.c2p(*i), color=colors[j]) for i,j in zip(ifs.S, ifs.trans_used)]

        dots = ifs_to_dots(fern, ax)
        dots_before = 3
        groups = divide_dots(dots, dots_before)

        self.wait(0.5)
        self.play(
            AnimationGroup(
                *[FadeIn(d) for d in groups],
                lag_ratio=1,
                run_time=10
            ),
        )

        self.wait(1)

from PIL import Image as PILImage

class PixelUpdatedImageMobject(ImageMobject):
    def __init__(self, image_path, **kwargs):
        super().__init__(image_path, **kwargs)
        # Load the image using Pillow (PIL)
        self.image = PILImage.open(image_path)
        self.pixels = np.array(self.image)  # Convert image to a numpy array of pixels
        self.width, self.height = self.image.size

    def update_pixelwise(self, modification_function):
        """
        This method updates the image pixels.

        :param modification_function: A function that takes and returns a numpy array of pixels.
        """
        # Apply the modification function to the current pixel data
        self.pixels = modification_function(self.pixels)

        # Convert the updated numpy array back to a PIL image
        updated_image = PILImage.fromarray(self.pixels)

        # Update the image object with the modified image
        self.set_image(updated_image)

    def get_image_array(self):
        """Returns the current pixel array."""
        return self.pixels

# # Example Scene
# class PixelUpdateScene(Scene):
#     def construct(self):
#         # Create the image object
#         image_path = "path/to/your/image.png"  # Replace with a valid image path
#         img = PixelUpdatedImageMobject(image_path)

#         # Display the image
#         self.add(img)

#         # Define a pixel modification function
#         def modify_pixels(pixels):
#             # Example: Invert the colors of the image
#             return 255 - pixels  # Simple inversion for demonstration

#         # Update the image pixels over time
#         self.play(
#             UpdateFromFunc(img, lambda m: m.update_pixelwise(modify_pixels)),
#             run_time=2
#         )

#         self.wait(1)


class ImageIFS(Scene):
    def construct(self):
        colors = [RED, BLUE, YELLOW, PURPLE]
        ax = Axes(x_range=(-2.7, 2.7), y_range=(0, 10), x_length=8, y_length=10).to_edge(DOWN)
        Dot.set_default(radius=0.03, color=WHITE)
        def ifs_to_dots(ifs) -> list[Dot]:
            return [Dot(ax.c2p(*i), color=colors[j]) for i,j in zip(ifs.S, ifs.trans_used)]


        fern = ifs.IFS_coral()
        fern.iterate(1_000_000)
        # dots = ifs_to_dots(fern)
        # dots_before = 3
        # groups = divide_dots(dots, dots_before)

        # self.wait(0.5)
        # self.play(
        #     AnimationGroup(
        #         *[FadeIn(d) for d in groups],
        #         lag_ratio=1,
        #         run_time=10
        #     ),
        # )

        self.wait(1)

        n = 2**10
        imageArray = np.uint8(
            [[BLACK.to_int_rgb() for i in range(0, n)] for _ in range(0, n)]
        )
        for pos, c in zip(fern.S, fern.trans_used):
            color = colors[c]
            pixel = int((pos[0] + 5) / 11 * n), int((pos[1] / 11 * n))
            imageArray[pixel] = color.to_int_rgb()

        image = ImageMobject(imageArray)

        # def update_image(image):
        #     fern.iterate()
        #     pos = fern.S[-1]
        #     color = colors[fern.trans_used[-1]]
        #     pixel = int((pos[0] + 5) / 10 * n), int((pos[1]/10 * n))
        #     image.pixel_array[pixel] = color.to_int_rgb()

        # image.add_updater(update_image)
        self.add(image)
        self.wait(2)
        # image.clear_updaters()
        # for i in fern.S:



# %manim -p -ql --disable_caching -v Warning Fern
