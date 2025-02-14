package com.optiflow.domain;

import java.util.List;

import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import com.optiflow.dto.ElectricityCostPredictItemDto;

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

@Getter
@Setter
@ToString
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "electricity_cost_predict")
public class ElectricityCostPredict {
	
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@Column(name = "cost_id")
	private int costId;
	
    @Column(name = "datetime")
    private String datetime;
    
    @Column(name = "truth")
    @Lob
    @JdbcTypeCode(SqlTypes.JSON)
    private List<ElectricityCostPredictItemDto> truth;
    
    @Column(name = "optimization")
    @Lob
    @JdbcTypeCode(SqlTypes.JSON)
    private List<ElectricityCostPredictItemDto> optimization;
    
	@ManyToOne
	@JoinColumn(name = "reservoir_id")
	private Reservoir reservoirId;

}
