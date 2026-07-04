package utils

func NormalizeLabels(labels []int, numLabels int) [][]float64 {
	labelArr := make([][]float64, len(labels))
	for index, label := range labels {
		row := make([]float64, numLabels)
		for i := range row {
			row[i] = 0
		}
		row[label] = 1.0
		labelArr[index] = row

	}
	return labelArr
}

type ImageNormalizer struct {
	minVal      float64
	maxVal      float64
	targetMin   float64
	targetMax   float64
	valRange    float64
	targetRange float64
}

func NewImageNormalizer(minVal, maxVal, targetMin, targetMax float64) ImageNormalizer {
	return ImageNormalizer{
		minVal:      minVal,
		maxVal:      maxVal,
		targetMin:   targetMin,
		targetMax:   targetMax,
		valRange:    maxVal - minVal,
		targetRange: targetMax - targetMin,
	}
}

func (normalizer *ImageNormalizer) NormalizeImages(images [][]int) [][]float64 {

	normalizedImages := make([][]float64, len(images))

	for i, image := range images {
		newImage := make([]float64, len(image))
		for j, val := range image {
			newImage[j] = normalizer.normalizedVal(val)
		}
		normalizedImages[i] = newImage
	}
	return normalizedImages
}

func (normalizer *ImageNormalizer) normalizedVal(val int) float64 {
	return normalizer.targetMin + (normalizer.targetRange)*(float64(val)-normalizer.minVal)/(normalizer.valRange)
}
