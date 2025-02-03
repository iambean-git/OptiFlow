package com.optiflow.controller;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.optiflow.domain.Reservoir;
import com.optiflow.domain.ReservoirData;
import com.optiflow.dto.ReservoirStats;
import com.optiflow.persistence.ReservoirRepository;
import com.optiflow.service.ReservoirDataService;

@RestController
@RequestMapping("/api/reservoirdata")
public class ReservoirDataController {

	@Autowired
	private ReservoirDataService reservoirDataService;
	@Autowired
	private ReservoirRepository reservoirRepo;
	
	@GetMapping("/{datetime}")
	public ResponseEntity<List<ReservoirData>> findByObservationTime(@PathVariable String datetime){
		LocalDateTime localDateTime = LocalDateTime.parse(datetime);
		List<ReservoirData> datas = reservoirDataService.findByObservationTime(localDateTime);
		return ResponseEntity.ok(datas);
	}

	@GetMapping("/daily/{reservoirId}")
	public ResponseEntity<List<ReservoirStats>> getDailyStatsByReservoirId(@PathVariable int reservoirId) {
		List<ReservoirStats> stats = reservoirDataService.findDailyStatsByReservoirId(reservoirId);
		return new ResponseEntity<>(stats, HttpStatus.OK);
	}

	 // 시간별 통계 (일별 기준)
    @GetMapping("/hourly/{date}/{reservoirName}") // 경로 변경: /hourly/daily
    public ResponseEntity<List<ReservoirStats>> findHourlyStatsByDailyObservationTimeAndReservoirId(
            @PathVariable String date, // YYYY-MM-DD 형식 (일별 기준)
            @PathVariable String reservoirName) {
    	
		Optional<Reservoir> reservoirOptional = reservoirRepo.findByName(reservoirName); // 이름으로 Reservoir 조회
        if (!reservoirOptional.isPresent()) { // Reservoir가 없을 경우 404 에러 반환
            return new ResponseEntity<>(HttpStatus.NOT_FOUND); // 또는 다른 에러 처리 방식 적용 가능
        }
        int reservoirId = reservoirOptional.get().getReservoirId(); // Reservoir 엔티티에서 reservoirId 추출
        List<ReservoirStats> stats = reservoirDataService.findHourlyStatsByDailyObservationTimeAndReservoirId(date, reservoirId);
        return new ResponseEntity<>(stats, HttpStatus.OK);
    }

    // 일별 통계 (월별 기준)
    @GetMapping("/daily/{month}/{reservoirName}") // 경로 변경: /daily/monthly
    public ResponseEntity<List<ReservoirStats>> findDailyStatsByMonthlyObservationTimeAndReservoirId(
            @PathVariable String month, // YYYY-MM 형식 (월별 기준)
            @PathVariable String reservoirName) {
    	
		Optional<Reservoir> reservoirOptional = reservoirRepo.findByName(reservoirName); // 이름으로 Reservoir 조회
        if (!reservoirOptional.isPresent()) { // Reservoir가 없을 경우 404 에러 반환
            return new ResponseEntity<>(HttpStatus.NOT_FOUND); // 또는 다른 에러 처리 방식 적용 가능
        }
        int reservoirId = reservoirOptional.get().getReservoirId(); // Reservoir 엔티티에서 reservoirId 추출
        List<ReservoirStats> stats = reservoirDataService.findDailyStatsByMonthlyObservationTimeAndReservoirId(month, reservoirId);
        return new ResponseEntity<>(stats, HttpStatus.OK);
    }

    // 월별 통계 (년별 기준)
    @GetMapping("/monthly/{year}/{reservoirName}") // 경로 변경: /monthly/yearly
    public ResponseEntity<List<ReservoirStats>> findMonthlyStatsByYearlyObservationTimeAndReservoirId(
            @PathVariable String year, // YYYY 형식 (년별 기준)
            @PathVariable String reservoirName) {
    	
		Optional<Reservoir> reservoirOptional = reservoirRepo.findByName(reservoirName); // 이름으로 Reservoir 조회
        if (!reservoirOptional.isPresent()) { // Reservoir가 없을 경우 404 에러 반환
            return new ResponseEntity<>(HttpStatus.NOT_FOUND); // 또는 다른 에러 처리 방식 적용 가능
        }
        int reservoirId = reservoirOptional.get().getReservoirId(); // Reservoir 엔티티에서 reservoirId 추출
        List<ReservoirStats> stats = reservoirDataService.findMonthlyStatsByYearlyObservationTimeAndReservoirId(year, reservoirId);
        return new ResponseEntity<>(stats, HttpStatus.OK);
    }
}
