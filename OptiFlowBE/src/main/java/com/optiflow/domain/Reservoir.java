package com.optiflow.domain;

import java.util.Date;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.Temporal;
import jakarta.persistence.TemporalType;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@Getter @Setter @ToString
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "reservoir")
public class Reservoir {
	@Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "reservoir_id")
    private int reservoirId;

    @Column(name = "name")
    private String name;
    
    @Column(name = "location")
    private String location;
    
    @Column(name = "max_height")
    private Float height;
    
    @Temporal(TemporalType.TIMESTAMP)
    @Column(name = "last_updated")
	private Date lastUpdated = new Date();
    
    @Column(name = "capacity")
    private Float capacity;
}
