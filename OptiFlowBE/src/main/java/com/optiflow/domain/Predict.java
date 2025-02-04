package com.optiflow.domain;

import java.util.Date;
import java.util.List;

import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Lob;
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
@Table(name = "predict")
public class Predict {
	
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "predict_id")
    private int predictId;

    @Column(name = "text")
    private String text;
    
    @Column(name = "result")
    @Lob // Large Object (TEXT, BLOB 등) 타입 지정
    @JdbcTypeCode(SqlTypes.JSON) // JSON 타입으로 지정 (MySQL 8+ 또는 PostgreSQL)
    private List<String> result; // List<String> 타입으로 변경 (JSON으로 직렬화/역직렬화)
    
    @Temporal(TemporalType.TIMESTAMP)
    @Column(name = "created_at")
	private Date createdDate = new Date();
}
