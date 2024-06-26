#include "raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void computeMandelbrot(int *output, float *wavelength, float xmin, float xmax, float ymin, float ymax, int width, int height, int maxIter, float minWaveLength, float maxWaveLength);
extern void calcRGB(double len, unsigned char* rgb);

// TODO Bring all state relevant properties into a struct

void saveState(int *output, int width, int height) {
  FILE *fp = fopen("distribution.txt", "w");
  if (fp != NULL) {
    for (int i = 0; i < width * height; i++) {
      fprintf(fp, "%d\n", output[i]);
    }
    fclose(fp);
  }
}

Color getColor (float wavelength) {
  unsigned char rgb[3];
  calcRGB(wavelength, rgb); 
  Color color = { };
  color = (Color){rgb[0], rgb[1], rgb[2], 255};
  return color;
}

int main() {

  int width = 2560 * 1.0;
  int height = 1600 * 0.9 * 1.0;

  InitWindow(width, height, "Mandelbrot Set");

  BeginDrawing();
  ClearBackground(RAYWHITE);
  EndDrawing();

  float xcenter = -0.75; 
  float ycenter = 0.0;   
  float xrange = 3.5; 

  // float xcenter = 0.275 ;
  // float ycenter = 0.0	 ;   
  // float xrange =   0.05; 

  // float xcenter = -0.747  ;
  // float ycenter = 0.065	 ;   
  // float xrange =   0.02; 

  // float xcenter = -1.25066  ;
  // float ycenter = 0.02012	 ;   
  // float xrange =   0.0004; 

  // float xcenter = -1.3107  ;
  // float ycenter = 0.0038	 ;   
  // float xrange =   0.3; 

  // float xcenter = -0.724  ;
  // float ycenter = 0.246	 ;   
  // float xrange =   0.01; 

  float aspectRatio = (float)width / (float)height;
  float yrange = xrange / aspectRatio;

  float xmin = xcenter - xrange / 2;
  float xmax = xcenter + xrange / 2;

  float ymin = ycenter - yrange / 2;
  float ymax = ycenter + yrange / 2;

  int maxIter = 1000;
  int *output = (int *)malloc(width * height * sizeof(int));
  float *wavelengths = (float *)malloc(width * height * sizeof(float));
  float minWaveLength = 400;
  float maxWaveLength = 700;

  computeMandelbrot(output, wavelengths,
                    xmin, xmax,
                    ymin, ymax,
                    width, height, 
                    maxIter, 
                    minWaveLength, maxWaveLength);


  SetTargetFPS(60);

  bool is_drawn = 0;

  while (!WindowShouldClose()) {
    BeginDrawing();

    if (is_drawn == 0) {
      ClearBackground(BLACK);
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          int idx = y * width + x;
          float wavelength = wavelengths[idx];
          Color color = (output [idx]==maxIter)? BLACK : getColor(wavelength);
          DrawPixel(x, y, color);
        }
      }
      saveState(output, width, height);
      is_drawn = 1;
    }

    EndDrawing();
  }

  free(output); 
  free(wavelengths); 
  CloseWindow(); 

  return 0;

}











