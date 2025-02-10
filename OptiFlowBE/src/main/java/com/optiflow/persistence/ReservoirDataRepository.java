package com.optiflow.persistence;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import com.optiflow.domain.Reservoir;
import com.optiflow.domain.ReservoirData;
import com.optiflow.domain.ReservoirDataId;

@Repository
public interface ReservoirDataRepository extends JpaRepository<ReservoirData,ReservoirDataId> {
	
	// 특정 복합키로 height 값만 가져오기
    @Query("SELECT r.height FROM ReservoirData r WHERE r.reservoirId.id = :reservoirId AND r.observationTime = :observationTime")
    Float findHeightByReservoirIdAndObservationTime(
            @Param("reservoirId") Integer reservoirId,
            @Param("observationTime") LocalDateTime observationTime);
    
	List<ReservoirData> findByObservationTime(LocalDateTime observationTime);
	List<ReservoirData> findByReservoirIdAndObservationTimeBetween(Reservoir reservoirId, LocalDateTime startTime, LocalDateTime endTime);
	@Query(value = "SELECT " +
            "DATE_FORMAT(rd.observation_time, '%Y-%m-%d') AS observationTime, " +
            "r.reservoir_id AS reservoirId, " +
            "SUM(rd.input) AS totalInput, " +
            "SUM(rd.output) AS totalOutput, " +
            "AVG(rd.height) AS avgHeight " +
            "FROM reservoir_data rd " +
            "JOIN reservoir r ON rd.reservoir_id = r.reservoir_id " +
            "WHERE r.reservoir_id = :reservoirId " +
            "GROUP BY DATE_FORMAT(rd.observation_time, '%Y-%m-%d'), r.reservoir_id " +
            "ORDER BY DATE_FORMAT(rd.observation_time, '%Y-%m-%d')",
            nativeQuery = true)
	List<Object[]> findDailyStatsByReservoirId(@Param("reservoirId") int reservoirId);

    // 시간별 통계 (일별 기준)
    @Query(value = "SELECT " +
            "DATE_FORMAT(rd.observation_time, '%Y-%m-%d %H') AS observationTime, " +
            "r.reservoir_id AS reservoirId, " +
            "SUM(rd.input) AS totalInput, " +
            "SUM(rd.output) AS totalOutput, " +
            "AVG(rd.height) AS avgHeight " +
            "FROM reservoir_data rd " +
            "JOIN reservoir r ON rd.reservoir_id = r.reservoir_id " +
            "WHERE r.reservoir_id = :reservoirId " +
            "AND rd.observation_time BETWEEN :startTime AND :endTime " + // 기간 검색 (시간별)
            "GROUP BY DATE_FORMAT(rd.observation_time, '%Y-%m-%d %H'), r.reservoir_id " +
            "ORDER BY DATE_FORMAT(rd.observation_time, '%Y-%m-%d %H')",
            nativeQuery = true)
    List<Object[]> findHourlyStatsByDailyObservationTimeAndReservoirId(
            @Param("startTime") String startTime,
            @Param("endTime") String endTime,
            @Param("reservoirId") int reservoirId);


    // 일별 통계 (월별 기준)
    @Query(value = "SELECT " +
            "DATE_FORMAT(rd.observation_time, '%Y-%m-%d') AS observationTime, " +
            "r.reservoir_id AS reservoirId, " +
            "SUM(rd.input) AS totalInput, " +
            "SUM(rd.output) AS totalOutput, " +
            "AVG(rd.height) AS avgHeight " +
            "FROM reservoir_data rd " +
            "JOIN reservoir r ON rd.reservoir_id = r.reservoir_id " +
            "WHERE r.reservoir_id = :reservoirId " +
            "AND rd.observation_time BETWEEN :startTime AND :endTime " +
            "GROUP BY DATE_FORMAT(rd.observation_time, '%Y-%m-%d'), r.reservoir_id " +
            "ORDER BY DATE_FORMAT(rd.observation_time, '%Y-%m-%d')",
            nativeQuery = true)
    List<Object[]> findDailyStatsByMonthlyObservationTimeAndReservoirId(
            @Param("startTime") String startTime,
            @Param("endTime") String endTime,
            @Param("reservoirId") int reservoirId);


    // 월별 통계 (년별 기준)
    @Query(value = "SELECT " +
            "DATE_FORMAT(rd.observation_time, '%Y-%m') AS observationTime, " +
            "r.reservoir_id AS reservoirId, " +
            "SUM(rd.input) AS totalInput, " +
            "SUM(rd.output) AS totalOutput, " +
            "AVG(rd.height) AS avgHeight " +
            "FROM reservoir_data rd " +
            "JOIN reservoir r ON rd.reservoir_id = r.reservoir_id " +
            "WHERE r.reservoir_id = :reservoirId " +
            "AND rd.observation_time BETWEEN :startTime AND :endTime " +
            "GROUP BY DATE_FORMAT(rd.observation_time, '%Y-%m'), r.reservoir_id " +
            "ORDER BY DATE_FORMAT(rd.observation_time, '%Y-%m')",
            nativeQuery = true)
    List<Object[]> findMonthlyStatsByYearlyObservationTimeAndReservoirId(
            @Param("startTime") String startTime, 
            @Param("endTime") String endTime,
            @Param("reservoirId") int reservoirId);
}