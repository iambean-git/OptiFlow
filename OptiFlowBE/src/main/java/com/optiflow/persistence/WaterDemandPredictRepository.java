package com.optiflow.persistence;

import java.util.List;
import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.optiflow.domain.Reservoir;
import com.optiflow.domain.WaterDemandPredict;

@Repository
public interface WaterDemandPredictRepository extends JpaRepository<WaterDemandPredict, Integer> {

	Optional<WaterDemandPredict> findByDatetime(String datetime);
	Optional<WaterDemandPredict> findByDatetimeAndReservoirIdAndUsedModel(String datetime, Reservoir reservoirId, String modelName);
	List<WaterDemandPredict> findByReservoirIdAndDatetimeBetweenOrderByDatetimeAsc(Reservoir reservoirId,
			String startDate, String endDate);
}
