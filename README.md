# Realtime-Manuscript-Auto-Camera-Scanner
A gentle try to replicate the famous app CamScanner from zero but with half of the capabilities and double of problems.

A video can be found in my [Linkedln page](https://www.linkedin.com/in/diego-bonilla-salvador/).

I've approached the problem from a pixel-threshold perspective using the HSV colorspace to detect the range of values a sheet of paper can arange.
Then some postprocessing using kernel morphology transformations to remove external crap and a canny filter to improve the contour detection capabilities.

Et voilà!

