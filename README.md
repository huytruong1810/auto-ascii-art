# auto-ascii-art

1. Access your computer's camera to collect an image
2. Mask select objects of interest detected by Mask-RCNN
3. Turn those masked objects into ASCII format via pixel density mapping
4. Show the resulting ASCII art

 These steps are repeated on the series of images collected by your computer's camera
 Limitations: Mask-RCNN has a long inference time. Therefore, the outputs seem glitchy when done in real time.
