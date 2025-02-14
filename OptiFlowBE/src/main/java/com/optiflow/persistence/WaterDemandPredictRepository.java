package com.optiflow.persistence;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.optiflow.domain.WaterDemandPredict;
import com.optiflow.domain.Reservoir;

@Repository
public interface WaterDemandPredictRepository extends JpaRepository<WaterDemandPredict, Integer> {

	Optional<WaterDemandPredict> findByDatetime(String datetime);
	Optional<WaterDemandPredict> findByDatetimeAndReservoirIdAndUsedModel(String datetime, Reservoir reservoirId, String modelName);
}
