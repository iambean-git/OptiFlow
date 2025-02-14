package com.optiflow.dto;

import lombok.Data;
import lombok.Getter;
import lombok.Setter;

@Data
@Getter @Setter
public class WaterDemandPredictRequestDto {
	private String name;
	private String modelName;
    private String datetime;
    private Float waterLevel;
}
