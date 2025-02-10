package com.optiflow.dto;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonProperty;

import lombok.Data;
import lombok.Getter;
import lombok.Setter;

@Data
@Getter @Setter
public class PredictResponseDto {
	
	@JsonProperty("prediction") // JSON 키 "prediction" 과 매핑
	private List<PredictionItemDto> prediction;
	
	@JsonProperty("optiflow")
	private List<PredictionItemDto> optiflow;

	public PredictResponseDto() {
		this.prediction = java.util.Collections.emptyList();
		this.optiflow = java.util.Collections.emptyList();
	}
	
}
