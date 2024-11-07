// Fast Mandelbrot Rendering with GPU in C#.
// Guy Fernando - i4cy (2024)

using System.Numerics;

using ILGPU;
using ILGPU.Runtime;

namespace Mandelbrot;

public static class MandelbrotKernel
{
    public static void ComputeMandelbrotFrame(
        Index1D index, ArrayView1D<int, Stride1D.Dense> output,
        double centerX, double centerY, double scale, short width, short height)
    {
        int x = index % width;
        int y = index / width;

        // Calculate aspect ratio and scaling factors.
        double aspectRatio = (double)width / height;
        double adjustedScaleX, adjustedScaleY;
        if (aspectRatio >= 1.0)
        {
            adjustedScaleX = scale * aspectRatio;
            adjustedScaleY = scale;
        }
        else
        {
            adjustedScaleX = scale;
            adjustedScaleY = scale / aspectRatio;
        }

        // Calculate the complex coordinate.
        Complex c = GetComplexCoordinate(x, y, adjustedScaleX, adjustedScaleY, centerX, centerY, width, height);

        // Perform Mandelbrot iteration.
        short iterations = CalculateMandelbrotPixel(c);

        // Write result to output.
        output[index] = iterations;
    }

    private static Complex GetComplexCoordinate(
        int x, int y, double scaleX, double scaleY, double centerX, double centerY, short width, short height)
    {
        double real = (x * scaleX / width) - (scaleX / 2) + centerX;
        double imaginary = (y * scaleY / height) - (scaleY / 2) + centerY;

        return new Complex(real, imaginary);
    }

    private static short CalculateMandelbrotPixel(Complex c)
    {
        Complex z = Complex.Zero;
        short iterations = 0;

        while (iterations < MandelbrotConstants.MaxIterations && (z.Real * z.Real + z.Imaginary * z.Imaginary) <= 4.0)
        {
            // z based on Mandelbrot iteration formula z = z^2 + c.
            z = z * z + c;
            iterations++;
        }

        return iterations;
    }
}
