use sptorch_hal::serial::{
    Matmul32x32Command, ResultRead32Command, ResultValue32Payload, ResultWindowStatusPayload,
    ResultWindowStatusReadCommand, ScratchRead32Command, ScratchValue32Payload, ScratchWrite32Command, SerialFrame,
    SerialOpcode, SerialStatusCode, SerialStatusPayload, SerialStreamDecoder, MATMUL32X32_FLAG_CLEAR_OUTPUT,
    MATMUL32X32_FLAG_LAST_K_TILE,
};

const PING_FRAME_GOLDEN: &[u8] = &[
    0x53, 0x50, 0x01, 0x01, 0x04, 0x03, 0x02, 0x01, 0x03, 0x00, 0x00, 0x00, 0xa5, 0x00, 0x00, 0x00, 0xaa, 0xbb, 0xcc,
    0x7b, 0x12, 0x1d, 0x47, 0x00,
];

const STATUS_BUSY_DETAIL_GOLDEN: &[u8] = &[0x04, 0x00, 0x00, 0x00, 0x44, 0x33, 0x22, 0x11];

const ACK_BUSY_FRAME_GOLDEN: &[u8] = &[
    0x53, 0x50, 0x01, 0x7e, 0x07, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00,
    0x00, 0x44, 0x33, 0x22, 0x11, 0x6e, 0xe6, 0xdc, 0x44, 0x00, 0x00, 0x00, 0x00,
];

const MATMUL32_PAYLOAD_GOLDEN: &[u8] = &[
    0x04, 0x03, 0x02, 0x01, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x28, 0x27, 0x26, 0x25, 0x24, 0x23, 0x22,
    0x21, 0x38, 0x37, 0x36, 0x35, 0x34, 0x33, 0x32, 0x31, 0x05, 0x00, 0x00, 0x00,
];

const MATMUL32_FRAME_GOLDEN: &[u8] = &[
    0x53, 0x50, 0x01, 0x10, 0x09, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x03, 0x02,
    0x01, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x28, 0x27, 0x26, 0x25, 0x24, 0x23, 0x22, 0x21, 0x38, 0x37,
    0x36, 0x35, 0x34, 0x33, 0x32, 0x31, 0x05, 0x00, 0x00, 0x00, 0x67, 0x00, 0x75, 0x6c, 0x00, 0x00, 0x00, 0x00,
];

const SCRATCH_WRITE32_PAYLOAD_GOLDEN: &[u8] = &[0x44, 0x00, 0x00, 0x00, 0x44, 0x33, 0x22, 0x11];
const SCRATCH_WRITE32_FRAME_GOLDEN: &[u8] = &[
    0x53, 0x50, 0x01, 0x20, 0x0a, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x00, 0x00,
    0x00, 0x44, 0x33, 0x22, 0x11, 0xc1, 0xfc, 0x65, 0x21, 0x00, 0x00, 0x00, 0x00,
];
const SCRATCH_ACK_FRAME_GOLDEN: &[u8] = &[
    0x53, 0x50, 0x01, 0x7e, 0x0a, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x23, 0x50, 0xbe, 0x87, 0x00, 0x00, 0x00, 0x00,
];
const SCRATCH_READ32_PAYLOAD_GOLDEN: &[u8] = &[0x44, 0x00, 0x00, 0x00];
const SCRATCH_READ32_FRAME_GOLDEN: &[u8] = &[
    0x53, 0x50, 0x01, 0x21, 0x0b, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x00, 0x00,
    0x00, 0x97, 0x1e, 0xe5, 0xda,
];
const SCRATCH_VALUE32_PAYLOAD_GOLDEN: &[u8] = &[0x44, 0x00, 0x00, 0x00, 0x44, 0x33, 0x22, 0x11];
const SCRATCH_VALUE32_FRAME_GOLDEN: &[u8] = &[
    0x53, 0x50, 0x01, 0x22, 0x0b, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x00, 0x00,
    0x00, 0x44, 0x33, 0x22, 0x11, 0x86, 0x82, 0xc3, 0xd0, 0x00, 0x00, 0x00, 0x00,
];

const RESULT_READ32_PAYLOAD_GOLDEN: &[u8] = &[0x38, 0x37, 0x36, 0x35];
const RESULT_READ32_FRAME_GOLDEN: &[u8] = &[
    0x53, 0x50, 0x01, 0x30, 0x0c, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x38, 0x37, 0x36,
    0x35, 0x07, 0x6b, 0xac, 0x7e,
];
const RESULT_VALUE32_PAYLOAD_GOLDEN: &[u8] = &[0x38, 0x37, 0x36, 0x35, 0x0d, 0x07, 0x06, 0x05];
const RESULT_VALUE32_FRAME_GOLDEN: &[u8] = &[
    0x53, 0x50, 0x01, 0x31, 0x0c, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x38, 0x37, 0x36,
    0x35, 0x0d, 0x07, 0x06, 0x05, 0xc9, 0x2a, 0xb3, 0x89, 0x00, 0x00, 0x00, 0x00,
];
const RESULT_WINDOW_STATUS_READ_FRAME_GOLDEN: &[u8] = &[
    0x53, 0x50, 0x01, 0x32, 0x0d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xa8, 0x56, 0x42,
    0x10, 0x00, 0x00, 0x00, 0x00,
];
const RESULT_WINDOW_STATUS_PAYLOAD_GOLDEN: &[u8] = &[
    0x01, 0x04, 0x04, 0x00, 0x00, 0x20, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];
const RESULT_WINDOW_STATUS_FRAME_GOLDEN: &[u8] = &[
    0x53, 0x50, 0x01, 0x33, 0x0d, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x04, 0x04,
    0x00, 0x00, 0x20, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xea, 0xb4, 0x05, 0xfe, 0x00, 0x00,
    0x00, 0x00,
];

#[test]
fn ping_frame_matches_wire_golden_vector() {
    let frame = SerialFrame::with_flags(SerialOpcode::Ping, 0x0102_0304, 0x00a5, vec![0xaa, 0xbb, 0xcc]);

    assert_eq!(frame.encode().unwrap(), PING_FRAME_GOLDEN);
    assert_eq!(SerialFrame::decode(PING_FRAME_GOLDEN).unwrap(), frame);
}

#[test]
fn status_payload_and_ack_frame_match_wire_golden_vector() {
    let payload = SerialStatusPayload {
        code: SerialStatusCode::Busy,
        detail: 0x1122_3344,
    };

    assert_eq!(payload.encode(), STATUS_BUSY_DETAIL_GOLDEN);
    let frame = SerialFrame::new(SerialOpcode::Ack, 7, payload.encode());
    assert_eq!(frame.encode().unwrap(), ACK_BUSY_FRAME_GOLDEN);
    assert_eq!(
        SerialStatusPayload::decode(&SerialFrame::decode(ACK_BUSY_FRAME_GOLDEN).unwrap().payload).unwrap(),
        payload
    );
}

#[test]
fn matmul32_command_and_frame_match_wire_golden_vector() {
    let command = Matmul32x32Command {
        tile_id: 0x0102_0304,
        a_offset: 0x1112_1314_1516_1718,
        b_offset: 0x2122_2324_2526_2728,
        out_offset: 0x3132_3334_3536_3738,
        flags: MATMUL32X32_FLAG_CLEAR_OUTPUT | MATMUL32X32_FLAG_LAST_K_TILE,
    };

    assert_eq!(command.encode_payload(), MATMUL32_PAYLOAD_GOLDEN);
    let frame = command.into_frame(9);
    assert_eq!(frame.encode().unwrap(), MATMUL32_FRAME_GOLDEN);
    assert_eq!(
        Matmul32x32Command::decode_payload(&SerialFrame::decode(MATMUL32_FRAME_GOLDEN).unwrap().payload).unwrap(),
        command
    );
}

#[test]
fn scratch32_commands_and_frames_match_wire_golden_vectors() {
    let write_command = ScratchWrite32Command::new(0x44, 0x1122_3344);
    assert_eq!(write_command.encode_payload(), SCRATCH_WRITE32_PAYLOAD_GOLDEN);
    let write_frame = write_command.into_frame(10);
    assert_eq!(write_frame.encode().unwrap(), SCRATCH_WRITE32_FRAME_GOLDEN);
    assert_eq!(
        ScratchWrite32Command::decode_payload(&SerialFrame::decode(SCRATCH_WRITE32_FRAME_GOLDEN).unwrap().payload)
            .unwrap(),
        write_command
    );

    let ack_frame = SerialFrame::ack(10, SerialStatusPayload::ok());
    assert_eq!(ack_frame.encode().unwrap(), SCRATCH_ACK_FRAME_GOLDEN);

    let read_command = ScratchRead32Command::new(0x44);
    assert_eq!(read_command.encode_payload(), SCRATCH_READ32_PAYLOAD_GOLDEN);
    let read_frame = read_command.into_frame(11);
    assert_eq!(read_frame.encode().unwrap(), SCRATCH_READ32_FRAME_GOLDEN);
    assert_eq!(
        ScratchRead32Command::decode_payload(&SerialFrame::decode(SCRATCH_READ32_FRAME_GOLDEN).unwrap().payload)
            .unwrap(),
        read_command
    );

    let value_payload = ScratchValue32Payload::new(0x44, 0x1122_3344);
    assert_eq!(value_payload.encode_payload(), SCRATCH_VALUE32_PAYLOAD_GOLDEN);
    let value_frame = value_payload.into_frame(11);
    assert_eq!(value_frame.encode().unwrap(), SCRATCH_VALUE32_FRAME_GOLDEN);
    assert_eq!(
        ScratchValue32Payload::decode_payload(&SerialFrame::decode(SCRATCH_VALUE32_FRAME_GOLDEN).unwrap().payload)
            .unwrap(),
        value_payload
    );
}

#[test]
fn result32_commands_and_frames_match_wire_golden_vectors() {
    let read_command = ResultRead32Command::new(0x3536_3738);
    assert_eq!(read_command.encode_payload(), RESULT_READ32_PAYLOAD_GOLDEN);
    let read_frame = read_command.into_frame(12);
    assert_eq!(read_frame.encode().unwrap(), RESULT_READ32_FRAME_GOLDEN);
    assert_eq!(
        ResultRead32Command::decode_payload(&SerialFrame::decode(RESULT_READ32_FRAME_GOLDEN).unwrap().payload).unwrap(),
        read_command
    );

    let value_payload = ResultValue32Payload::new(0x3536_3738, 0x0506_070d);
    assert_eq!(value_payload.encode_payload(), RESULT_VALUE32_PAYLOAD_GOLDEN);
    let value_frame = value_payload.into_frame(12);
    assert_eq!(value_frame.encode().unwrap(), RESULT_VALUE32_FRAME_GOLDEN);
    assert_eq!(
        ResultValue32Payload::decode_payload(&SerialFrame::decode(RESULT_VALUE32_FRAME_GOLDEN).unwrap().payload)
            .unwrap(),
        value_payload
    );
}

#[test]
fn result_window_status_commands_and_frames_match_wire_golden_vectors() {
    let read_frame = ResultWindowStatusReadCommand.into_frame(13);
    assert_eq!(read_frame.encode().unwrap(), RESULT_WINDOW_STATUS_READ_FRAME_GOLDEN);
    assert_eq!(
        ResultWindowStatusReadCommand::decode_payload(
            &SerialFrame::decode(RESULT_WINDOW_STATUS_READ_FRAME_GOLDEN)
                .unwrap()
                .payload
        )
        .unwrap(),
        ResultWindowStatusReadCommand
    );

    let status_payload = ResultWindowStatusPayload::new(true, 4, 4, 0x2000, 4);
    assert_eq!(status_payload.encode_payload(), RESULT_WINDOW_STATUS_PAYLOAD_GOLDEN);
    let status_frame = status_payload.into_frame(13);
    assert_eq!(status_frame.encode().unwrap(), RESULT_WINDOW_STATUS_FRAME_GOLDEN);
    assert_eq!(
        ResultWindowStatusPayload::decode_payload(
            &SerialFrame::decode(RESULT_WINDOW_STATUS_FRAME_GOLDEN).unwrap().payload
        )
        .unwrap(),
        status_payload
    );
}

#[test]
fn stream_decoder_accepts_golden_vectors_with_chunk_boundaries() {
    let mut stream = vec![0x00, 0xff];
    stream.extend_from_slice(PING_FRAME_GOLDEN);
    stream.extend_from_slice(ACK_BUSY_FRAME_GOLDEN);
    stream.extend_from_slice(MATMUL32_FRAME_GOLDEN);
    stream.extend_from_slice(SCRATCH_WRITE32_FRAME_GOLDEN);
    stream.extend_from_slice(SCRATCH_ACK_FRAME_GOLDEN);
    stream.extend_from_slice(SCRATCH_READ32_FRAME_GOLDEN);
    stream.extend_from_slice(SCRATCH_VALUE32_FRAME_GOLDEN);
    stream.extend_from_slice(RESULT_READ32_FRAME_GOLDEN);
    stream.extend_from_slice(RESULT_VALUE32_FRAME_GOLDEN);
    stream.extend_from_slice(RESULT_WINDOW_STATUS_READ_FRAME_GOLDEN);
    stream.extend_from_slice(RESULT_WINDOW_STATUS_FRAME_GOLDEN);

    let mut decoder = SerialStreamDecoder::new();
    assert!(decoder.push_bytes(&stream[..3]).unwrap().is_empty());
    let frames = decoder.push_bytes(&stream[3..]).unwrap();

    assert_eq!(frames.len(), 11);
    assert_eq!(frames[0].opcode, SerialOpcode::Ping);
    assert_eq!(frames[1].opcode, SerialOpcode::Ack);
    assert_eq!(frames[2].opcode, SerialOpcode::Matmul32x32);
    assert_eq!(frames[3].opcode, SerialOpcode::ScratchWrite32);
    assert_eq!(frames[4].opcode, SerialOpcode::Ack);
    assert_eq!(frames[5].opcode, SerialOpcode::ScratchRead32);
    assert_eq!(frames[6].opcode, SerialOpcode::ScratchValue32);
    assert_eq!(frames[7].opcode, SerialOpcode::ResultRead32);
    assert_eq!(frames[8].opcode, SerialOpcode::ResultValue32);
    assert_eq!(frames[9].opcode, SerialOpcode::ResultWindowStatusRead);
    assert_eq!(frames[10].opcode, SerialOpcode::ResultWindowStatus);
}
