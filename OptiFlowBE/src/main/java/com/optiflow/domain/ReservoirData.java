package com.optiflow.domain;

import java.time.LocalDateTime;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.IdClass;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@Getter @Setter @ToString
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "reservoir_data")
@IdClass(ReservoirDataId.class)
public class ReservoirData {
	
	@Id
    @Column(name = "observation_time")
    private LocalDateTime observationTime;
	
	@Id
	@ManyToOne
	@JoinColumn(name = "reservoir_id", insertable = false, updatable = false)
	private Reservoir reservoirId;

	@Column(name = "input")
	private Float input;
	
    @Column(name = "output")
    private Float output;

    @Column(name = "height")
    private Float height;

}
