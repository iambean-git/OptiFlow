package com.optiflow.persistence;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.optiflow.domain.Predict;
import com.optiflow.domain.Reservoir;

@Repository
public interface PredictRepository extends JpaRepository<Predict, Integer> {

	Optional<Predict> findByDatetime(String datetime);
	Optional<Predict> findByDatetimeAndReservoirIdAndUsedModel(String datetime, Reservoir reservoirId, String modelName);
}
