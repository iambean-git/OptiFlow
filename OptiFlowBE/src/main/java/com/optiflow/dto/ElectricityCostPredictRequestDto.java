package com.optiflow.dto;

import lombok.Data;
import lombok.Getter;
import lombok.Setter;

@Data
@Getter @Setter
public class ElectricityCostPredictRequestDto {
	private String name;
    private String datetime;
    private Float waterLevel;
}
