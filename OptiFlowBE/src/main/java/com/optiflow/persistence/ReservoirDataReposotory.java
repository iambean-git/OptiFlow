package com.optiflow.persistence;

import java.time.LocalDateTime;
import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import com.optiflow.domain.ReservoirData;

@Repository
public interface ReservoirDataReposotory extends JpaRepository<ReservoirData, LocalDateTime> {
	
	//  @Query(value = "SELECT " + 생성자 필요한데 쓰면 안됨 왜냐면 jpql 형식인데 mysql에서는 못 알아먹기 때문
	//  "DATE_FORMAT(observation_time, '%Y') AS observationTime, " +
	//  "r.reservoir_id AS reservoirId, " +
	//  "SUM(input) AS totalInput, " +
	//  "SUM(output) AS totalOutput, " +
	//  "AVG(rd.height) AS height " +
	//  "FROM reservoir_data rd " +
	//  "JOIN reservoir r ON rd.reservoir_id = r.reservoir_id " +
	//  "WHERE r.reservoir_id = :reservoirId " +
	//  "GROUP BY DATE_FORMAT(observation_time, '%Y'), r.reservoir_id " + // 수정됨
	//  "ORDER BY DATE_FORMAT(observation_time, '%Y')", nativeQuery = true) // 수정됨
	//List<ReservoirStats> findYearlyStatsByReservoirId(@Param("reservoirId") Long reservoirId);
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
//	@Query(value = "SELECT YEAR(r.observationTime) AS year, " +
//            "MONTH(r.observationTime) AS month, " +
//            "SUM(r.input) AS total_input, " +
//            "SUM(r.output) AS total_output " +
//            "AVG(r.height) AS height " +
//            "FROM ReservoirData r " +
//            "WHERE YEAR(r.observationTime) = :nyun " + 2023을 적었는데도 202로 알아먹는지는 의문
//            "GROUP BY YEAR(r.observationTime), MONTH(r.observationTime) " +
//            "ORDER BY month",
//            nativeQuery = true)
//	List<Object[]> findReservoirStatsByYear(@Param("nyun") int nyun);
	
	@Query(value = "SELECT " +
            "DATE_FORMAT(rd.observation_time, '%Y-%m') AS observationTime, " +
            "r.reservoir_id AS reservoirId, " +
            "SUM(rd.input) AS totalInput, " +
            "SUM(rd.output) AS totalOutput, " +
            "AVG(rd.height) AS avgHeight " +
            "FROM reservoir_data rd " +
            "JOIN reservoir r ON rd.reservoir_id = r.reservoir_id " +
            "WHERE r.reservoir_id = :reservoirId " +
            "GROUP BY DATE_FORMAT(rd.observation_time, '%Y-%m'), r.reservoir_id " +
            "ORDER BY DATE_FORMAT(rd.observation_time, '%Y-%m')",
            nativeQuery = true)
	List<Object[]> findMonthlyStatsByReservoirId(@Param("reservoirId") int reservoirId);


//	@Query("SELECT" + FUNCTION과 DATE_FORMAT은 MySQL인데 value, nativcequery 안적어서 그런듯
//           "FUNCTION('DATE_FORMAT', r.observationTime, :pattern), " +
//           "r.reservoirId, " +
//           "SUM(r.input), SUM(r.output), AVG(r.height)) " +
//           "FROM ReservoirData r " +
//           "GROUP BY FUNCTION('DATE_FORMAT', r.observationTime, :pattern), r.reservoirId")
//    List<Object> getGroupedReservoirStats(@Param("pattern") String pattern);
//

}