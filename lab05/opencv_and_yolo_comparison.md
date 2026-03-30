# Comparison between OpenCV and YOLOv8 for bird counting

| name | Count | Accuracy | Notes |
|---|---|---|---|
OpenCV | 14/16 | 87.5% | Overcounted and imagined birds in background. But performed perfectly for less crowded images.
YOLOv8 | 0/16 | 0% | Failed to detect any birds in the images. Detected "possible" planes and a toilet.

## Comparison images:

> Toilet and Car come from a fallback to broader classes when nothing is detected with flying classes in Yolo, like: plane, bird, kite, etc.

![Comparison images](markdown/1.jpg)
> OpenCV: 4/4 Correct <br>
> Yolo: 0/4

![Comparison images](markdown/2.jpg)
> OpenCV: 7/7 Correct <br>
> Yolo: 0/7 - 4 False Positives (2 planes, 2 toilets)

![Comparison images](markdown/3.jpg)
> OpenCV: 6/6 Correct <br>
> Yolo: 0/6 - 1 False positive (car)

![Comparison images](markdown/4.jpg)
> OpenCV: 2/2 Correct <br>
> Yolo: 0/2

![Comparison images](markdown/5.jpg)
> OpenCV: 11/20 Correct - with 9 False Positives and birds clustered as one<br>
> Yolo: 0/2

![Comparison images](markdown/6.jpg)
> OpenCV: 5/5 Correct <br>
> Yolo: 0/5

![Comparison images](markdown/7.jpg)
> OpenCV: 1/1 Correct <br>
> Yolo: 0/1

![Comparison images](markdown/8.jpg)
> OpenCV: 1/1 Correct <br>
> Yolo: 0/1

![Comparison images](markdown/9.jpg)
> OpenCV: 2/2 Correct <br>
> Yolo: 0/2

![Comparison images](markdown/10.jpg)
> OpenCV: 2/2 Correct <br>
> Yolo: 0/2

![Comparison images](markdown/11.jpg)
> OpenCV: 2/2 Correct <br>
> Yolo: 0/2

![Comparison images](markdown/12.jpg)
> OpenCV: 4/4 Correct <br>
> Yolo: 0/4 - 1 False Positive (plane)

![Comparison images](markdown/13.jpg)
> OpenCV: 9/9 Correct <br>
> Yolo: 0/9

![Comparison images](markdown/14.jpg)
> OpenCV: 5/15-17 Correct <br>
> Yolo: 0/15-17

> [!WARNING]  
> I struggled to count these ones. There's a lot of shadow that is not background, so it has to be a bird, but is also not defined enough to be picked up by the detectors.

![Comparison images](markdown/15.jpg)
> OpenCV: 22/23 Correct - 2 False Positives. Red box means uncounted, though I marked it as correct. <br>
> Yolo: 0/2

> [!NOTE]  
> Red box means not accepted, but marked. That's due to the area. It's either too small or too large. In this case it's because it's right at the edge of the image and after erosion the blob becomes too small. This was meant as indicator for "pepper filter like" effects for manual checks, but in this case it fails it's job.

![Comparison images](markdown/16.jpg)
> OpenCV: 2/2 Correct <br>
> Yolo: 0/2