package com.optiflow.dto;

import lombok.Data;
import lombok.Getter;
import lombok.Setter;

@Data
@Getter @Setter
public class PredictRequestDto {
	private String name;
	private String modelName;
    private String datetime;
    private Float waterLevel;
}
