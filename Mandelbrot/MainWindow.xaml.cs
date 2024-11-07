// Fast Mandelbrot Rendering with GPU in C#.
// Guy Fernando - i4cy (2024)

using System.Globalization;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace Mandelbrot;

public static class MandelbrotConstants
{
    public const short MaxIterations = 1000;
}

public partial class MainWindow : Window
{
    private short width;
    private short height;

    private double centerX = -0.74;
    private double centerY = 0.15;
    private double scale = 2.5;

    private bool isPanning = false;
    private bool isZooming = false;
    private Point startPanPoint;

    private Context context;
    private Accelerator accelerator;
    private Action<Index1D, ArrayView1D<int, Stride1D.Dense>, double, double, double, short, short> kernel;

    public MainWindow()
    {
        InitializeComponent();

        // Initialize ILGPU context and accelerator.
        context = Context.Create(builder => builder.Cuda());
        accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

        // Load the kernel once during initialization.
        kernel = accelerator.LoadAutoGroupedStreamKernel
            <Index1D, ArrayView1D<int, Stride1D.Dense>, double, double, double, short, short>(MandelbrotKernel.ComputeMandelbrotFrame);

        // Add event handlers for zooming, panning, and resizing.
        this.MouseWheel += MainWindow_MouseWheel;
        this.MouseRightButtonDown += MainWindow_MouseRightButtonDown;
        this.MouseLeftButtonDown += MainWindow_MouseLeftButtonDown;
        this.MouseLeftButtonUp += MainWindow_MouseLeftButtonUp;
        this.MouseMove += MainWindow_MouseMove;
        this.SizeChanged += MainWindow_SizeChanged;
        this.KeyDown += MainWindow_KeyDown;

        // Generate the initial Mandelbrot set.
        GenerateMandelbrotFrame();
    }

    private void MainWindow_MouseRightButtonDown(object sender, MouseButtonEventArgs e)
    {
        // Reset zoom and panning.
        centerX = -0.5;
        centerY = 0.0;
        scale = 3.5;

        // Regenerate the Mandelbrot set with the updated dimensions.
        GenerateMandelbrotFrame();
    }

    private void MainWindow_SizeChanged(object sender, SizeChangedEventArgs e)
    {
        // Update the width and height based on the new window size.
        width = (short)e.NewSize.Width;
        height = (short)e.NewSize.Height;

        // Regenerate the Mandelbrot set with the updated dimensions.
        GenerateMandelbrotFrame();
    }

    private void MainWindow_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        // Start panning mode.
        isPanning = true;
        startPanPoint = e.GetPosition(MandelbrotImage);
        MandelbrotImage.CaptureMouse();
    }

    private void MainWindow_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
    {
        // End panning mode.
        isPanning = false;
        MandelbrotImage.ReleaseMouseCapture();
    }

    private void MainWindow_MouseMove(object sender, MouseEventArgs e)
    {
        if (isPanning)
        {
            // Calculate the movement delta.
            Point currentPoint = e.GetPosition(MandelbrotImage);
            double deltaX = (currentPoint.X - startPanPoint.X) / width * scale;
            double deltaY = (currentPoint.Y - startPanPoint.Y) / height * scale;

            // Adjust the center point based on the movement.
            centerX -= deltaX;
            centerY -= deltaY;

            // Update the start point for the next movement calculation.
            startPanPoint = currentPoint;

            // Regenerate the Mandelbrot set with the new parameters.
            GenerateMandelbrotFrame();
        }
    }

    private void MainWindow_MouseWheel(object sender, MouseWheelEventArgs e)
    {
        // Get the mouse position relative to the image.
        Point mousePos = e.GetPosition(MandelbrotImage);

        // Normalize mouse position to [-1, 1] range in the complex plane.
        double normX = (mousePos.X / width - 0.5) * scale;
        double normY = (mousePos.Y / height - 0.5) * scale;

        // Adjust scale based on the scroll direction
        scale *= e.Delta > 0 ? 0.9 : 1.1;

        // Adjust the center point based on the normalized mouse position.
        centerX += normX * (1 - scale / (scale * (e.Delta > 0 ? 0.9 : 1.1)));
        centerY += normY * (1 - scale / (scale * (e.Delta > 0 ? 0.9 : 1.1)));

        // Regenerate the Mandelbrot set with the new parameters.
        GenerateMandelbrotFrame();
    }

    private void MainWindow_KeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Space)
        {
            if (!isZooming)
            {
                isZooming = true;
                StartAutoZoom();
            }
            else
            {
                isZooming = false;
            }
        }
    }

    private async void StartAutoZoom()
    {
        // Set fixed coordinates for auto-zoom, near a point of interest.
        centerX = -0.74335165531181;
        centerY = +0.13138323820835;

        // Zoom speed multiplier.
        const double zoomFactorIncrement = 0.95;

        while (isZooming && scale > 1e-13) // Stop when zoom factor is extremely high.
        {
            // Reduce the zoom scale.
            scale *= zoomFactorIncrement;

            // Render the Mandelbrot set at the new zoom level.
            GenerateMandelbrotFrame();

            // Allow the UI to update by awaiting a small delay ensuring UI responsiveness.
            await Task.Delay(1);
        }
    }

    private void UpdateStatusBar()
    {
        CenterXText.Text = $"Center X: {centerX:F14}";
        CenterYText.Text = $"Center Y: {centerY:F14}";

        // Display zoom factor in engineering format
        string zoomFormatted = (1 / scale).ToString("F1", CultureInfo.InvariantCulture);
        ZoomFactorText.Text = $"Zoom: {zoomFormatted}";
    }

    protected override void OnClosed(EventArgs e)
    {
        // Cleanup resources on window close.
        base.OnClosed(e);
        accelerator.Dispose();
        context.Dispose();
    }

    private void GenerateMandelbrotFrame()
    {
        if (width <= 0 || height <= 0)
            return; // Skip rendering if dimensions are invalid.

        // Update Status Bar.
        UpdateStatusBar();

        // Calculate the aspect ratio
        double aspectRatio = (double)width / height;

        // Determine the scaling factors to maintain aspect ratio.
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

        // Allocate memory on the GPU.
        using var buffer = accelerator.Allocate1D<int>(width * height);

        // Execute the kernel with the current parameters.
        kernel((int)(width * height), buffer.View, centerX, centerY, scale, width, height);
        accelerator.Synchronize();

        // Retrieve the results from GPU
        int[] result = buffer.GetAsArray1D();

        // Set the Image control source to display the Mandelbrot set.
        MandelbrotImage.Source = CreateFrameBitmap(result);
    }

    private WriteableBitmap CreateFrameBitmap(int[] pixels)
    {
        // Create a WriteableBitmap and fill it with the Mandelbrot set image.
        WriteableBitmap bitmap = new WriteableBitmap(width, height, 96, 96, PixelFormats.Bgra32, null);
        bitmap.Lock();
        unsafe
        {
            IntPtr pBackBuffer = bitmap.BackBuffer;
            for (short y = 0; y < height; y++)
            {
                for (short x = 0; x < width; x++)
                {
                    Color color = GetPixelColor(pixels[y * width + x]);

                    *((uint*)pBackBuffer + y * width + x) =
                        (uint)((color.A << 24) | (color.R << 16) | (color.G << 8) | (color.B << 0));
                }
            }
        }
        bitmap.AddDirtyRect(new Int32Rect(0, 0, width, height));
        bitmap.Unlock();

        return bitmap;
    }

    private static Color GetPixelColor(int iterations)
    {
        if (iterations >= MandelbrotConstants.MaxIterations)
        {
            return Colors.Black;
        }
        else
        {
            // Convert HSV to RGB for a more colour pleasing image.
            return ColorFromHSV(
                ((double)(iterations)) / MandelbrotConstants.MaxIterations * 360.0,
                1.0,
                1.0
                );
        }
    }

    public static Color ColorFromHSV(double hue, double saturation, double value)
    {
        sbyte hi = Convert.ToSByte(Math.Floor(hue / 60) % 6);
        double f = hue / 60 - Math.Floor(hue / 60);

        value = value * 255;
        byte v = Convert.ToByte(value);
        byte p = Convert.ToByte(value * (1 - saturation));
        byte q = Convert.ToByte(value * (1 - f * saturation));
        byte t = Convert.ToByte(value * (1 - (1 - f) * saturation));

        if (hi == 0)
            return Color.FromArgb(255, v, t, p);
        else if (hi == 1)
            return Color.FromArgb(255, q, v, p);
        else if (hi == 2)
            return Color.FromArgb(255, p, v, t);
        else if (hi == 3)
            return Color.FromArgb(255, p, q, v);
        else if (hi == 4)
            return Color.FromArgb(255, t, p, v);
        else
            return Color.FromArgb(255, v, p, q);
    }
}
