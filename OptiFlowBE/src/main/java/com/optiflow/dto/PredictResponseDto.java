package com.optiflow.dto;

import java.util.List;

import lombok.Data;

@Data
public class PredictResponseDto {
    private List<String> result;
    
    public PredictResponseDto() {
    	this.result = java.util.Collections.emptyList();
    }
}
