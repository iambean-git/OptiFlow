package com.optiflow.domain;

import java.util.List;

import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import com.optiflow.dto.WaterDemandPredictionItemDto;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.Lob;
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
@Table(name = "water_demand_predict")
public class WaterDemandPredict {
	
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "demand_id")
    private int demandId;

    @Column(name = "datetime")
    private String datetime;
    
    @Column(name = "used_model")
    private String usedModel;
    
    @Column(name = "prediction")
    @Lob // Large Object (TEXT, BLOB 등) 타입 지정
    @JdbcTypeCode(SqlTypes.JSON) // JSON 타입으로 지정
    private List<WaterDemandPredictionItemDto> prediction;
    
    @Column(name = "optiflow")
    @Lob
    @JdbcTypeCode(SqlTypes.JSON)
    private List<WaterDemandPredictionItemDto> optiflow;
    
	@ManyToOne
	@JoinColumn(name = "reservoir_id")
	private Reservoir reservoirId;
}
