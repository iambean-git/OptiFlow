package com.optiflow.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@Getter @Setter
@AllArgsConstructor
public class ReservoirStats {
	private String observationTime;
    private int reservoirId;
    private Double totalInput;
    private Double totalOutput;
    private Double height;
}
