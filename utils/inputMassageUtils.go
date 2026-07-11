package utils

import "fmt"

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
	inputMin    float64
	inputMax    float64
	targetMin   float64
	targetMax   float64
	inputRange  float64
	targetRange float64
}

func NewImageNormalizer(inputMin, inputMax, targetMin, targetMax float64) ImageNormalizer {
	// fmt.Printf("MinVal: %.2f, MaxVal: %.2ff, TargetMin: %.2f, targetMax: %.2f, InputRange: %.2f, targetRange: %.2f\n", inputMin, inputMax, targetMin, targetMax, inputMax-inputMin, targetMax-targetMin)
	return ImageNormalizer{
		inputMin:    inputMin,
		inputMax:    inputMax,
		targetMin:   targetMin,
		targetMax:   targetMax,
		inputRange:  inputMax - inputMin,
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
	fmt.Printf("MinVal: %.2f, TargetMin: %.2f,  InputRange: %.2f, targetRange: %.2f, val: %d\n", normalizer.inputMin, normalizer.targetMin, normalizer.inputRange, normalizer.targetRange, val)
	return normalizer.targetMin + (normalizer.targetRange)*(float64(val)-normalizer.inputMin)/(normalizer.inputRange)
}
