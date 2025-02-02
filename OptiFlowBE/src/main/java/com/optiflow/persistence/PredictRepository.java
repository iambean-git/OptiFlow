package com.optiflow.persistence;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.optiflow.domain.Predict;

@Repository
public interface PredictRepository extends JpaRepository<Predict, Integer> {

}
