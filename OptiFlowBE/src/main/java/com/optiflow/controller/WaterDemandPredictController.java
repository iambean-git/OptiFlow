package com.optiflow.controller;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.optiflow.domain.WaterDemandPredict;
import com.optiflow.domain.Reservoir;
import com.optiflow.dto.WaterDemandPredictRequestDto;
import com.optiflow.dto.WaterDemandPredictResponseDto;
import com.optiflow.persistence.ReservoirDataRepository;
import com.optiflow.persistence.ReservoirRepository;
import com.optiflow.service.WaterDemandPredictService;

import io.swagger.v3.oas.annotations.tags.Tag;

@RestController
@RequestMapping("/api")
@Tag(name = "WaterDemandPredict API", description = "물 수요량 예측 관련 API")
public class WaterDemandPredictController {

	private static final Logger log = LoggerFactory.getLogger(WaterDemandPredictController.class);

	@Autowired
	private WaterDemandPredictService waterService;

	@Autowired
	private ReservoirRepository reservoirRepo;

	@Autowired
	private ReservoirDataRepository reservoirDataRepo;

	@GetMapping("/results")
	public ResponseEntity<List<WaterDemandPredict>> getAllPredicts() {
		List<WaterDemandPredict> predictList = waterService.getAllPredicts();
		return ResponseEntity.ok(predictList);
	}

	@GetMapping("/predict/{modelName}/{reservoirName}/{datetime}")
	public ResponseEntity<Map<String, List<?>>> getPrediction(@PathVariable String modelName,
			@PathVariable String reservoirName, @PathVariable String datetime) {
		log.info("Received prediction request with datetime: {}", datetime);
		
		if (!reservoirName.equalsIgnoreCase("j") && !reservoirName.equalsIgnoreCase("d")
				&& !reservoirName.equalsIgnoreCase("l")) {
			reservoirName = "j";
		}
		
		Optional<Reservoir> reservoirOptional = reservoirRepo.findByName(reservoirName);
		Float reservoirArea = reservoirOptional.get().getArea();
		LocalDateTime localDateTime = LocalDateTime.parse(datetime);
		int reservoirId = reservoirOptional.get().getReservoirId();
		Float height = reservoirDataRepo.findHeightByReservoirIdAndObservationTime(reservoirId, localDateTime);
		Float waterLevel = reservoirArea * height;
		WaterDemandPredictRequestDto requestDto = new WaterDemandPredictRequestDto();

		requestDto.setName(reservoirName);
		requestDto.setModelName(modelName);
		requestDto.setDatetime(datetime);
		requestDto.setWaterLevel(waterLevel);
		WaterDemandPredictResponseDto responseDto = waterService.getPrediction(requestDto);
		List<WaterDemandPredictResponseDto> datas = new ArrayList<>();
		datas.add(responseDto);

		Map<String, List<?>> responseMap = waterService.convertToResponseMap(datas);
		return ResponseEntity.ok(responseMap);
	}
	
	@GetMapping("/dailywater/{name}/{datetime}")
    public ResponseEntity<Map<String, List<?>>> getDailyCost(
            @PathVariable String name,
            @PathVariable String datetime) {
        Map<String, List<?>> responseMap = waterService.getDailyCostData(name, datetime);
        return ResponseEntity.ok(responseMap);
    }
	
	@GetMapping("/monthlywater/{name}/{datetime}")
    public ResponseEntity<Map<String, List<?>>> getMonthlyCost(
            @PathVariable String name,
            @PathVariable String datetime) {
        Map<String, List<?>> responseMap = waterService.getMonthlyCostData(name, datetime);
        return ResponseEntity.ok(responseMap);
    }

}
