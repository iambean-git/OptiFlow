package com.optiflow.domain;

import java.time.LocalDateTime;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
@Entity
@Table(name = "member")
public class Member {

	@Id
	@Column(name = "member_id")
	private String username;

	@Column(name = "password")
	private String password;

	@Enumerated(EnumType.STRING)
	private Role role = Role.Role_User;

	@Column(name = "created_at")
	private LocalDateTime createdAt = LocalDateTime.now();

	@Column(name = "reservoir_id")
	private int reservoirId = 2;

	@Builder
	public Member(String username, String password, Role role) {
		this.username = username;
		this.password = password;
		this.role = role;
	}

	public enum Role {
		Role_Admin, Role_Manager, Role_User
	}
}

