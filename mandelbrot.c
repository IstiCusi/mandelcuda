#include "raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void computeMandelbrot(int *output, float xmin, float xmax, float ymin, float ymax, int width, int height, int maxIter);

int main() {
  
    InitWindow(0, 0, "Mandelbrot Set");
    int width = GetScreenWidth();
    int height = GetScreenHeight();

    float aspectRatio = (float)width / (float)height;
    int maxIter = 1000;
    int *output = (int *)malloc(width * height * sizeof(int));

    float xmin = -2.5;
    float xmax = 1.0;
    float xrange = xmax - xmin;
    float yrange = xrange / aspectRatio;
    float ymin = -yrange / 2;
    float ymax = yrange / 2;

    computeMandelbrot(output, xmin, xmax, ymin, ymax, width, height, maxIter);

    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(BLACK);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int iter = output[y * width + x];
                float normIter = (float)iter / maxIter;
                float hue = (normIter * 360.0f) - 90.0f;
                float saturation = 0.5f + 0.5f * sin(hue * 0.3);
                float brightness = 0.5f + 0.5f * cos(hue * 0.5);
                Color color = ColorFromHSV(hue, saturation, brightness);
                DrawPixel(x, y, color);
            }
        }

        EndDrawing();
    }

    free(output);
    CloseWindow();

    return 0;
}

