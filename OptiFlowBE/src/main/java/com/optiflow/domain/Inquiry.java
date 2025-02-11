package com.optiflow.domain;

import java.time.LocalDateTime;

import org.hibernate.annotations.CreationTimestamp;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import lombok.Data;

@Entity
@Data
@Table(name = "inquiry")
public class Inquiry {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "inquiry_id")
    private Long inquiryId;

    @Column(name = "name", nullable = false)
    private String name;

    @Column(name = "contact", nullable = false)
    private String contact;

    @Column(name = "email")
    private String email;

    @Column(name = "location")
    private String location;

    @Column(name = "tmpid")
    private String tmpid;

    @Column(name = "inquiry_details", nullable = false, columnDefinition = "TEXT")
    private String inquiryDetails;

    @CreationTimestamp
    @Column(name = "created_dt")
    private LocalDateTime createdDt;

    @Column(name = "staff_confirmed")
    private Boolean staffConfirmed = false;
}
