package com.optiflow.dto;

import lombok.Data;

@Data
public class PasswordChangeRequestDto {
	private String username;
	private String password;
	private String newpw;
}
