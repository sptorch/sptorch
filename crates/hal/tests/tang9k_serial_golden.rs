use sptorch_hal::serial::{
    Matmul32x32Command, SerialFrame, SerialOpcode, SerialStatusCode, SerialStatusPayload, SerialStreamDecoder,
    MATMUL32X32_FLAG_CLEAR_OUTPUT, MATMUL32X32_FLAG_LAST_K_TILE,
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
fn stream_decoder_accepts_golden_vectors_with_chunk_boundaries() {
    let mut stream = vec![0x00, 0xff];
    stream.extend_from_slice(PING_FRAME_GOLDEN);
    stream.extend_from_slice(ACK_BUSY_FRAME_GOLDEN);
    stream.extend_from_slice(MATMUL32_FRAME_GOLDEN);

    let mut decoder = SerialStreamDecoder::new();
    assert!(decoder.push_bytes(&stream[..3]).unwrap().is_empty());
    let frames = decoder.push_bytes(&stream[3..]).unwrap();

    assert_eq!(frames.len(), 3);
    assert_eq!(frames[0].opcode, SerialOpcode::Ping);
    assert_eq!(frames[1].opcode, SerialOpcode::Ack);
    assert_eq!(frames[2].opcode, SerialOpcode::Matmul32x32);
}
