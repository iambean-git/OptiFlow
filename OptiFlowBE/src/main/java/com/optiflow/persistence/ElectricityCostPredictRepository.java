package com.optiflow.persistence;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.optiflow.domain.ElectricityCostPredict;
import com.optiflow.domain.Reservoir;

@Repository
public interface ElectricityCostPredictRepository extends JpaRepository<ElectricityCostPredict, Integer> {
	Optional<ElectricityCostPredict> findByDatetime(String datetime);
	Optional<ElectricityCostPredict> findByDatetimeAndReservoirId(String datetime, Reservoir reservoirId);
}
