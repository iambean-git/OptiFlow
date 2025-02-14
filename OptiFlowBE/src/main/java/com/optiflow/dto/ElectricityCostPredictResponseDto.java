package com.optiflow.dto;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonProperty;

import lombok.Data;
import lombok.Getter;
import lombok.Setter;

@Data
@Getter @Setter
public class ElectricityCostPredictResponseDto {
	
	@JsonProperty("truth") // JSON 키 "prediction" 과 매핑
	private List<ElectricityCostPredictItemDto> truth;
	
	@JsonProperty("optimization") // JSON 키 "prediction" 과 매핑
	private List<ElectricityCostPredictItemDto> optimization;

	public ElectricityCostPredictResponseDto() {
		this.truth = java.util.Collections.emptyList();
		this.optimization = java.util.Collections.emptyList();
	}
}
