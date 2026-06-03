'use client';

import { FC, useRef } from 'react';
import { ThreeEvent, useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import * as THREE from 'three';
import { Sensor, SensorDetail } from '../lib/types';
import { SENSOR_3D_POSITIONS } from '../lib/constants';

const GATEWAY_POSITION: [number, number, number] = [0, 1.0, 0];

const sensorPosition = (id: number, total: number): [number, number, number] => {
    if (SENSOR_3D_POSITIONS[id]) {
        return SENSOR_3D_POSITIONS[id] as [number, number, number];
    }
    const columns = Math.ceil(Math.sqrt(total * 1.5));
    const rows = Math.ceil(total / columns);
    const col = id % columns;
    const row = Math.floor(id / columns);
    const xSpacing = 42 / Math.max(1, columns - 1);
    const zSpacing = 27 / Math.max(1, rows - 1);
    const jitter = ((id * 9301 + 49297) % 233280) / 233280 - 0.5;
    const x = -21 + col * xSpacing + jitter * Math.min(0.35, xSpacing * 0.25);
    const z = -13.5 + row * zSpacing - jitter * Math.min(0.35, zSpacing * 0.25);
    return [x, 0.6, z];
};

const sensorRadius = (total: number) => {
    if (total > 350) return 0.2;
    if (total > 180) return 0.28;
    if (total > 80) return 0.38;
    return 0.7;
};

const SensorNode: FC<{
    id: number;
    isOff: boolean;
    isAnomaly?: boolean;
    isShadow?: boolean;
    retrainActive?: boolean;
    position: [number, number, number];
    radius: number;
    showLabel: boolean;
    selected?: boolean;
    onSelect?: (id: number) => void;
}> = ({ id, isOff, isAnomaly = false, isShadow = false, retrainActive = false, position, radius, showLabel, selected = false, onSelect }) => {
    const meshRef = useRef<THREE.Mesh>(null);
    const onColor = isAnomaly ? '#ef4444' : '#3b82f6';
    const offColor = isAnomaly ? '#7f1d1d' : '#4b5563';
    const anomalyRingColor = '#ef4444';
    const emissiveIntensity = isOff ? 0 : 2.5;
    const handleClick = (event: ThreeEvent<MouseEvent>) => {
        event.stopPropagation();
        onSelect?.(id);
    };

    useFrame(({ clock }) => {
        if (!meshRef.current) return;
        if (!isAnomaly && !retrainActive) {
            meshRef.current.scale.setScalar(selected ? 1.22 : 1);
            return;
        }
        const pulse = 1 + Math.sin(clock.elapsedTime * (retrainActive ? 7 : 5) + id) * (retrainActive ? 0.12 : 0.08);
        meshRef.current.scale.setScalar((selected ? 1.22 : 1) * pulse);
    });

    return (
        <mesh
            ref={meshRef}
            position={position}
            castShadow
            onClick={handleClick}
            onPointerOver={(event) => {
                event.stopPropagation();
                document.body.style.cursor = 'pointer';
            }}
            onPointerOut={() => {
                document.body.style.cursor = 'default';
            }}
        >
            <sphereGeometry args={[radius, 16, 16]} />
            <meshStandardMaterial
                color={selected ? '#e0f2fe' : isOff ? offColor : onColor}
                emissive={selected ? '#38bdf8' : isOff ? (isAnomaly ? '#ef4444' : offColor) : onColor}
                emissiveIntensity={selected ? 2.8 : isAnomaly ? (isOff ? 0.75 : 2.8) : emissiveIntensity}
                roughness={0.4}
                metalness={0.2}
            />
            {isAnomaly && (
                <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -radius - 0.02, 0]}>
                    <ringGeometry args={[radius * 1.45, radius * 1.85, 32]} />
                    <meshBasicMaterial color={anomalyRingColor} transparent opacity={isOff ? 0.55 : 0.7} />
                </mesh>
            )}
            {isShadow && (
                <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -radius - 0.05, 0]}>
                    <ringGeometry args={[radius * 1.95, radius * 2.28, 32]} />
                    <meshBasicMaterial color="#60a5fa" transparent opacity={0.8} />
                </mesh>
            )}
            {showLabel && (
                <Text
                    position={[0, radius + 0.42, 0]}
                    color="white"
                    fontSize={Math.max(0.24, radius * 0.75)}
                    anchorX="center"
                    anchorY="middle"
                    outlineWidth={0.02}
                    outlineColor="black"
                >
                    {id + 1}
                </Text>
            )}
        </mesh>
    );
};

const GatewayNode: FC = () => {
    const headRef = useRef<THREE.Group>(null);

    useFrame(({ clock }) => {
        if (headRef.current) {
            headRef.current.rotation.y = clock.elapsedTime * 0.25;
        }
    });

    return (
        <group position={GATEWAY_POSITION}>
            <mesh position={[0, 1.15, 0]} castShadow>
                <cylinderGeometry args={[0.08, 0.12, 3.4, 16]} />
                <meshStandardMaterial color="#94a3b8" roughness={0.35} metalness={0.45} />
            </mesh>
            <group ref={headRef} position={[0, 3.0, 0]}>
                <mesh castShadow>
                    <octahedronGeometry args={[0.78, 1]} />
                    <meshStandardMaterial color="#facc15" emissive="#f59e0b" emissiveIntensity={1.25} roughness={0.25} metalness={0.3} />
                </mesh>
            </group>
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.45, 0]}>
                <ringGeometry args={[1.25, 1.45, 64]} />
                <meshBasicMaterial color="#facc15" transparent opacity={0.26} />
            </mesh>
            <Text
                position={[0, 4.0, 0]}
                color="white"
                fontSize={0.5}
                anchorX="center"
                anchorY="middle"
                outlineWidth={0.02}
                outlineColor="black"
            >
                FARM GATEWAY
            </Text>
        </group>
    );
};

export const RotatingSceneContent: FC<{
    sensors: Sensor[];
    sensorDetails?: SensorDetail[];
    groundTexture: THREE.CanvasTexture | null;
    selectedSensorId?: number | null;
    retrainActive?: boolean;
    onSelectSensor?: (id: number) => void;
}> = ({ sensors, sensorDetails = [], groundTexture, selectedSensorId = null, retrainActive = false, onSelectSensor }) => {
    const sceneGroupRef = useRef<THREE.Group>(null);
    const detailById = new Map(sensorDetails.map(detail => [detail.id, detail]));
    const totalSensors = sensors?.length || 0;
    const radius = sensorRadius(totalSensors);
    const showAllLabels = totalSensors <= 80;

    useFrame((state, delta) => {
        if (sceneGroupRef.current) {
            sceneGroupRef.current.rotation.y += delta * 0.05;
        }
    });

    return (
        <group ref={sceneGroupRef} position={[0, -1.0, 0]}>
            <group position={[0, -0.5, 0]}>
                <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
                    <planeGeometry args={[45, 30]} />
                    <meshStandardMaterial map={groundTexture} color={!groundTexture ? '#166534' : undefined} roughness={1} />
                </mesh>
                <mesh position={[0, -1, 15]}>
                    <boxGeometry args={[45, 2, 0.1]} />
                    <meshStandardMaterial color="#8d6e63" />
                </mesh>
                <mesh position={[0, -1, -15]}>
                    <boxGeometry args={[45, 2, 0.1]} />
                    <meshStandardMaterial color="#8d6e63" />
                </mesh>
                <mesh position={[22.5, -1, 0]} rotation={[0, -Math.PI / 2, 0]}>
                    <boxGeometry args={[30, 2, 0.1]} />
                    <meshStandardMaterial color="#8d6e63" />
                </mesh>
                <mesh position={[-22.5, -1, 0]} rotation={[0, Math.PI / 2, 0]}>
                    <boxGeometry args={[30, 2, 0.1]} />
                    <meshStandardMaterial color="#8d6e63" />
                </mesh>
                <mesh position={[0, -2, 0]} rotation={[-Math.PI / 2, 0, 0]}>
                    <planeGeometry args={[45, 30]} />
                    <meshStandardMaterial color="#5d4037" />
                </mesh>
            </group>
            <GatewayNode />
            {sensors?.filter(sensor => !sensor.is_off).slice(0, 100).map(sensor => {
                const position = sensorPosition(sensor.id, totalSensors);
                return (
                    <line key={`trace-${sensor.id}`}>
                        <bufferGeometry>
                            <bufferAttribute
                                attach="attributes-position"
                                args={[new Float32Array([
                                    GATEWAY_POSITION[0], GATEWAY_POSITION[1], GATEWAY_POSITION[2],
                                    position[0], position[1], position[2],
                                ]), 3]}
                            />
                        </bufferGeometry>
                        <lineBasicMaterial color="#38bdf8" transparent opacity={0.18} />
                    </line>
                );
            })}
            {sensors?.map(sensor => {
                const detail = detailById.get(sensor.id);
                const isAnomaly = detail?.is_anomaly || false;
                const isShadow = sensor.is_shadow || detail?.is_shadow || false;
                const selected = selectedSensorId === sensor.id;
                return (
                    <SensorNode
                        key={sensor.id}
                        id={sensor.id}
                        isOff={sensor.is_off}
                        isAnomaly={isAnomaly}
                        isShadow={isShadow}
                        retrainActive={retrainActive}
                        position={sensorPosition(sensor.id, totalSensors)}
                        radius={radius}
                        showLabel={showAllLabels || selected || isAnomaly}
                        selected={selected}
                        onSelect={onSelectSensor}
                    />
                );
            })}
        </group>
    );
};
