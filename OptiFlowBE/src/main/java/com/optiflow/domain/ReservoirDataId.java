package com.optiflow.domain;

import java.io.Serializable;
import java.time.LocalDateTime;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@EqualsAndHashCode
public class ReservoirDataId implements Serializable{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private Integer reservoirId;
    private LocalDateTime observationTime;

    // 기본 생성자 필요 (JPA 요구 사항)
    public ReservoirDataId() {}

    public ReservoirDataId(Integer reservoirId, LocalDateTime observationTime) {
        this.reservoirId = reservoirId;
        this.observationTime = observationTime;
    }
}
