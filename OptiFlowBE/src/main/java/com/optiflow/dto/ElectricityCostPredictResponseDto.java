package com.optiflow.dto;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonProperty;

import lombok.Data;
import lombok.Getter;
import lombok.Setter;

@Data
@Getter @Setter
public class ElectricityCostPredictResponseDto {
	
	@JsonProperty("truth") 
	private List<ElectricityCostPredictItemDto> truth;
	
	@JsonProperty("optimization")
	private List<ElectricityCostPredictItemDto> optimization;

	public ElectricityCostPredictResponseDto() {
		this.truth = java.util.Collections.emptyList();
		this.optimization = java.util.Collections.emptyList();
	}
}
